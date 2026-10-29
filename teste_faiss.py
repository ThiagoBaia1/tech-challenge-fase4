from pathlib import Path
import sys
import time

# Garante o acesso aos módulos em src/
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vectorstore import VectorStoreManager


def run_tests():
    print("=" * 65)
    print("  TESTE DE VALIDAÇÃO DO ÍNDICE VETORIAL (FAISS + METADADOS)")
    print("=" * 65)

    store = VectorStoreManager()

    if store.index is None or store.metadata_df is None:
        print("[-] Erro: O índice FAISS ou a tabela de metadados não foram encontrados.")
        print("    Execute primeiro o script: python -m scripts.build_index")
        return

    total_registos = store.index.ntotal
    print(f"\n[+] Total de documentos indexados: {total_registos:,}")

    # ---------------------------------------------------------
    # TESTE 1: Busca Semântica Pura (Sem correspondência exata de palavras)
    # ---------------------------------------------------------
    query_1 = "produto veio com defeito e quebrado no transporte"
    print("\n" + "-" * 65)
    print(f"[TESTE 1] Busca Semântica: '{query_1}'")
    print("-" * 65)

    t0 = time.time()
    res_1 = store.search(query=query_1, k=3)
    tempo_1 = time.time() - t0

    print(f"Recuperação concluída em {tempo_1:.4f} segundos:")
    for idx, (doc, meta, score) in enumerate(
        zip(res_1["documents"][0], res_1["metadatas"][0], res_1["distances"][0]),
        start=1,
    ):
        print(f"\n  Evidência #{idx} [Similaridade Cosseno: {score:.4f}]:")
        print(f"  - Avaliação ID: {meta.get('review_id')}")
        print(f"  - Nota do Cliente: {meta.get('score')} estrelas")
        print(f"  - Categoria: {meta.get('category')}")
        print(f"  - Texto: {doc}")

    # ---------------------------------------------------------
    # TESTE 2: Busca com Filtro de Metadados (Apenas avaliações com nota 5)
    # ---------------------------------------------------------
    query_2 = "chegou muito rápido antes do prazo previsto"
    filtro_nota_5 = {"score": 5}
    print("\n" + "-" * 65)
    print(f"[TESTE 2] Busca com Filtro: '{query_2}' (Filtro: score == 5)")
    print("-" * 65)

    t1 = time.time()
    res_2 = store.search(query=query_2, k=3, filter_metadata=filtro_nota_5)
    tempo_2 = time.time() - t1

    print(f"Recuperação concluída em {tempo_2:.4f} segundos:")
    for idx, (doc, meta, score) in enumerate(
        zip(res_2["documents"][0], res_2["metadatas"][0], res_2["distances"][0]),
        start=1,
    ):
        print(f"\n  Evidência #{idx} [Similaridade Cosseno: {score:.4f}]:")
        print(f"  - Avaliação ID: {meta.get('review_id')}")
        print(f"  - Nota do Cliente: {meta.get('score')} estrelas (Esperado: 5)")
        print(f"  - Houve Atraso: {meta.get('is_delayed')}")
        print(f"  - Texto: {doc}")

    # ---------------------------------------------------------
    # TESTE 3: Rastreabilidade das Evidências (Requisito do Edital)
    # ---------------------------------------------------------
    print("\n" + "-" * 65)
    print("[TESTE 3] Validação de Metadados Obrigatórios para Rastreabilidade")
    print("-" * 65)
    chaves_obrigatorias = {"review_id", "order_id", "score"}
    primeiro_meta = res_1["metadatas"][0][0] if res_1["metadatas"][0] else {}
    presentes = chaves_obrigatorias.issubset(primeiro_meta.keys())

    if presentes:
        print("[✓] Rastreabilidade garantida: review_id e order_id preservados.")
    else:
        print("[-] Atenção: Campos de rastreabilidade em falta nos metadados.")

    print("\n" + "=" * 65)
    print("  VALIDAÇÃO CONCLUÍDA")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()