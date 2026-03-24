"""
testar_recomendacao.py
----------------------
Recebe um id_client via argumento e retorna os produtos recomendados.

Uso:
    python src/testar_recomendacao.py --id_client 5
    python src/testar_recomendacao.py --id_client 5 --n 8
    python src/testar_recomendacao.py --id_client 5 --n 5 --verbose
"""

import argparse
import os
import sys
import sqlite3
import joblib
import pandas as pd
from scipy.sparse import csr_matrix

# Caminhos padrão
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(BASE_DIR, 'datasets', 'nautical.db')
MODEL_FILE  = os.path.join(BASE_DIR, 'models', 'modelo_recomendacao.pkl')

# Núcleo de recomendação
def recomendar_produtos(
    id_client: int,
    modelo,
    matriz: pd.DataFrame,
    df_produtos: pd.DataFrame,
    n_recomendacoes: int = 5,
    n_vizinhos: int = 10
) -> pd.DataFrame:
    """
    Retorna DataFrame com produtos recomendados para o cliente.

    Parâmetros
    ----------
    id_client        : ID do cliente
    modelo           : NearestNeighbors treinado
    matriz           : DataFrame cliente × produto (binário)
    df_produtos      : tabela de produtos
    n_recomendacoes  : quantas recomendações retornar
    n_vizinhos       : quantos vizinhos considerar

    Retorna
    -------
    DataFrame  [id_produto | nome | categoria | score]
    """
    if id_client not in matriz.index:
        raise ValueError(
            f"Cliente {id_client} não encontrado no modelo.\n"
            f"Clientes disponíveis: {sorted(matriz.index.tolist())}"
        )

    matriz_sparse = csr_matrix(matriz.values)
    idx_cliente   = matriz.index.get_loc(id_client)
    vetor_cliente = matriz_sparse[idx_cliente]

    distancias, indices = modelo.kneighbors(
        vetor_cliente,
        n_neighbors=min(n_vizinhos + 1, modelo.n_samples_fit_)
    )

    ids_vizinhos = [
        matriz.index[i] for i in indices[0]
        if matriz.index[i] != id_client
    ][:n_vizinhos]

    sims_vizinhos = [
        1 - d for i, d in zip(indices[0], distancias[0])
        if matriz.index[i] != id_client
    ][:n_vizinhos]

    ja_comprados = set(matriz.columns[matriz.loc[id_client] > 0])

    scores: dict = {}
    for vizinho, sim in zip(ids_vizinhos, sims_vizinhos):
        if sim <= 0:
            continue
        produtos_vizinho = set(matriz.columns[matriz.loc[vizinho] > 0])
        for prod in (produtos_vizinho - ja_comprados):
            scores[prod] = scores.get(prod, 0) + sim

    if not scores:
        return pd.DataFrame(columns=['id_produto', 'nome', 'categoria', 'score'])

    top = (
        pd.Series(scores, name='score')
        .sort_values(ascending=False)
        .head(n_recomendacoes)
        .reset_index()
    )
    top.columns = ['id_produto', 'score']

    top = top.merge(
        df_produtos.rename(columns={'code': 'id_produto', 'name': 'nome', 'category': 'categoria'}),
        on='id_produto', how='left'
    )

    return top[['id_produto', 'nome', 'categoria', 'score']]


def info_cliente_verbose(id_client: int, db_path: str) -> None:
    """Imprime perfil do cliente e histórico de compras."""
    conn = sqlite3.connect(db_path)
    cli = pd.read_sql(
        f"SELECT code, full_name, city, state FROM clientes WHERE code = {id_client}", conn
    )
    if cli.empty:
        print(f"[aviso] Cliente {id_client} não encontrado na tabela clientes.")
        conn.close()
        return

    c = cli.iloc[0]
    print(f"\n{'='*60}")
    print(f"  Cliente #{id_client}: {c['full_name']}")
    print(f"  Localização: {c['city']} / {c['state']}")

    historico = pd.read_sql(f"""
        SELECT p.name, p.category, COUNT(*) AS pedidos, SUM(v.total) AS receita
        FROM vendas v
        JOIN produtos p ON p.code = v.id_product
        WHERE v.id_client = {id_client}
        GROUP BY p.code
        ORDER BY receita DESC
    """, conn)

    print(f"\n  Histórico de compras ({len(historico)} produtos):")
    print(f"  {'Categoria':<15} {'Produto':<40} {'Pedidos':>7} {'Receita':>12}")
    print(f"  {'-'*15} {'-'*40} {'-'*7} {'-'*12}")
    for _, r in historico.iterrows():
        print(f"  {r['category']:<15} {r['name'][:40]:<40} {r['pedidos']:>7} R${r['receita']:>11,.2f}")
    print(f"{'='*60}")
    conn.close()

# Entry-point
def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendação — LH Nautical",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--id_client', type=int, required=True,
        help='ID do cliente para receber recomendações'
    )
    parser.add_argument(
        '--n', type=int, default=5,
        help='Número de produtos a recomendar (padrão: 5)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Exibe perfil completo do cliente e histórico de compras'
    )
    parser.add_argument(
        '--model', type=str, default=MODEL_FILE,
        help=f'Caminho para o arquivo .pkl do modelo\n(padrão: {MODEL_FILE})'
    )
    parser.add_argument(
        '--db', type=str, default=DB_PATH,
        help=f'Caminho para o nautical.db\n(padrão: {DB_PATH})'
    )

    args = parser.parse_args()

    # Validar arquivos
    if not os.path.exists(args.model):
        print(f"[erro] Modelo não encontrado: {args.model}")
        print("       Execute o notebook 05_modelo_recomendacao.ipynb primeiro.")
        sys.exit(1)

    if not os.path.exists(args.db):
        print(f"[erro] Banco de dados não encontrado: {args.db}")
        sys.exit(1)

    # Carregar modelo
    print(f"\nCarregando modelo de {args.model} ...")
    artefato    = joblib.load(args.model)
    modelo      = artefato['modelo']
    matriz      = artefato['matriz']
    df_produtos = artefato['df_produtos']

    # Info detalhada
    if args.verbose:
        info_cliente_verbose(args.id_client, args.db)

    # Gerar recomendações
    try:
        recomendacoes = recomendar_produtos(
            id_client       = args.id_client,
            modelo          = modelo,
            matriz          = matriz,
            df_produtos     = df_produtos,
            n_recomendacoes = args.n,
        )
    except ValueError as e:
        print(f"\n[erro] {e}")
        sys.exit(1)

    # Exibir resultado
    if recomendacoes.empty:
        print(f"\nNenhuma recomendação encontrada para o cliente {args.id_client}.")
        sys.exit(0)

    print(f"\n{'─'*60}")
    print(f"  Top {len(recomendacoes)} recomendações para o cliente #{args.id_client}")
    print(f"{'─'*60}")
    print(f"  {'#':<3} {'Categoria':<15} {'Produto':<40} {'Score':>6}")
    print(f"  {'─'*3} {'─'*15} {'─'*40} {'─'*6}")
    for rank, (_, row) in enumerate(recomendacoes.iterrows(), 1):
        print(
            f"  {rank:<3} {row['categoria']:<15} "
            f"{str(row['nome'])[:40]:<40} {row['score']:>6.3f}"
        )
    print(f"{'─'*60}\n")


if __name__ == '__main__':
    main()
