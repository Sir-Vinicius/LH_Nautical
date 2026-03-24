"""
testar_previsao.py
==================
Carrega o modelo treinado e gera uma previsão rápida de demanda.

Uso:
    python src/testar_previsao.py                        # previsão padrão (3 meses)
    python src/testar_previsao.py --meses 6              # previsão para 6 meses
    python src/testar_previsao.py --produto 62           # apenas produto id=62
    python src/testar_previsao.py --categoria ANCORAGEM  # apenas uma categoria
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Caminho padrão do modelo
DEFAULT_MODEL = Path(
    '/home/vinic/Documentos/Projetos/LH_Nautical/models/modelo_previsao.pkl'
)


# Helpers

def carregar_modelo(path: Path) -> dict:
    if not path.exists():
        print(f"[ERRO] Modelo não encontrado: {path}")
        print("       Execute primeiro o notebook 04_modelo_previsao.ipynb")
        sys.exit(1)
    with open(path, 'rb') as f:
        pacote = pickle.load(f)
    print(f"    Modelo carregado  (treinado em: {pacote.get('data_treino','?')})")
    print(f"    Origem : {pacote.get('db_origem','?')}")
    print(f"    Produtos disponíveis  : {len(pacote['resultados_produtos'])}")
    print(f"    Categorias disponíveis: {len(pacote['resultados_categorias'])}")
    return pacote


def prever(modelo, serie: pd.DataFrame, n_meses: int) -> pd.DataFrame:
    """Gera previsão para os próximos n_meses além do histórico."""
    ultima = serie['ds'].max()
    future = modelo.make_future_dataframe(periods=n_meses, freq='MS')
    fc     = modelo.predict(future)
    for col in ('yhat', 'yhat_lower', 'yhat_upper'):
        fc[col] = fc[col].clip(lower=0)
    return fc[fc['ds'] > ultima].head(n_meses)


def tabela(rows: list) -> str:
    if not rows:
        return "(sem resultados)"
    cols   = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    sep    = "  ".join("-" * widths[c] for c in cols)
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    lines  = [header, sep]
    for r in rows:
        lines.append("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


# Main

def main():
    parser = argparse.ArgumentParser(description="Previsão rápida de demanda")
    parser.add_argument('--meses',     type=int, default=3,
                        help='Meses à frente (padrão: 3)')
    parser.add_argument('--produto',   type=int, default=None,
                        help='ID do produto (opcional)')
    parser.add_argument('--categoria', type=str, default=None,
                        help='Nome da categoria (opcional)')
    parser.add_argument('--modelo',    type=str, default=str(DEFAULT_MODEL),
                        help='Caminho do arquivo .pkl')
    args = parser.parse_args()

    print("\n" + "═" * 64)
    print("  PREVISÃO DE DEMANDA — Inferência Rápida")
    print("═" * 64)

    pacote = carregar_modelo(Path(args.modelo))
    n      = args.meses

    # Produtos
    prods = pacote['resultados_produtos']
    if args.produto is not None:
        if args.produto not in prods:
            print(f"\n[AVISO] Produto {args.produto} não encontrado. "
                  f"IDs disponíveis: {list(prods.keys())}")
            prods = {}
        else:
            prods = {args.produto: prods[args.produto]}

    if args.categoria is None or args.produto is not None:
        rows = []
        for pid, res in prods.items():
            nome = res['label'].split('  [')[0][:35]
            fc   = prever(res['model'], res['serie'], n)
            for _, r in fc.iterrows():
                rows.append({
                    'ID':         pid,
                    'Produto':    nome,
                    'Mês':        r['ds'].strftime('%b/%Y'),
                    'Prev.(un.)': int(round(float(r['yhat']))),
                    'IC Inf.':    int(round(float(r['yhat_lower']))),
                    'IC Sup.':    int(round(float(r['yhat_upper']))),
                    'MAE±':       round(res['mae'], 1),
                })
        print(f"\n── Produtos (próximos {n} meses) " + "─" * 30)
        print(tabela(rows))

    # Categorias
    cats = pacote['resultados_categorias']
    if args.categoria is not None:
        cats = {k: v for k, v in cats.items()
                if args.categoria.upper() in k.upper()}
        if not cats:
            print(f"\n[AVISO] Categoria '{args.categoria}' não encontrada. "
                  f"Disponíveis: {list(pacote['resultados_categorias'].keys())}")

    if args.produto is None:
        rows = []
        for cat, res in cats.items():
            fc = prever(res['model'], res['serie'], n)
            for _, r in fc.iterrows():
                rows.append({
                    'Categoria':  cat,
                    'Mês':        r['ds'].strftime('%b/%Y'),
                    'Prev.(un.)': int(round(float(r['yhat']))),
                    'IC Inf.':    int(round(float(r['yhat_lower']))),
                    'IC Sup.':    int(round(float(r['yhat_upper']))),
                    'MAE±':       round(res['mae'], 1),
                })
        print(f"\n── Categorias (próximos {n} meses) " + "─" * 28)
        print(tabela(rows))

    print("\n" + "═" * 64)
    print("  Opções: --meses N  --produto ID  --categoria NOME")
    print("═" * 64 + "\n")


if __name__ == '__main__':
    main()
