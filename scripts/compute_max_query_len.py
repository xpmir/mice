from tests import get_max_query_length
import logging
from xpmir.text.huggingface.tokenizers import HFTokenizer
from format import aggregations

from logging_utils import setup_logging
setup_logging(level=logging.INFO)

if __name__ == '__main__':
    hf_id = "microsoft/MiniLM-L12-H384-uncased"

    # Initialize tokenizer
    tokenizer = HFTokenizer.C(model_id=hf_id)

    # Audit BEIR13 and LoTTE
    datasets_to_audit = aggregations["BEIR13"] + aggregations["Lotte-S"]

    # Map internal names to ir_datasets IDs where possible
    # This is a bit manual since we don't have the full mapping here
    # but get_max_query_length will try to prepare them.

    # Note: internal names like 'arguana' might need mapping to 'org.beir.arguana'
    # if they are not already in the EvaluationsCollection format.
    # For this script, let's use a subset of known IDs.
    beir_ids = [
        "org.beir.arguana",
        "org.beir.climate.fever",
        "org.beir.dbpedia.entity.test",
        "org.beir.fever.test",
        "org.beir.fiqa.test",
        "org.beir.hotpotqa.test",
        "org.beir.nfcorpus.test",
        "org.beir.nq",
        "org.beir.quora.test",
        "org.beir.scidocs",
        "org.beir.scifact.test",
        "org.beir.webis.touche2020.v2",
        "org.beir.trec.covid"
    ]

    lotte_ids = [
        "edu.stanford.lotte.writing.test.search",
        "edu.stanford.lotte.recreation.test.search",
        "edu.stanford.lotte.science.test.search",
        "edu.stanford.lotte.technology.test.search",
        "edu.stanford.lotte.lifestyle.test.search"
    ]

    # Check for query truncation
    max_query_lengths = get_max_query_length(tokenizer, beir_ids + lotte_ids)

    print("\n Query analysis for " + hf_id)
    print("\n" + "="*120)
    print(f"{'Dataset ID':<50} | {'Max':<5} | {'Smpl':<5} | {'Sample Query (truncated to 300 chars)'}")
    print("-" * 120)

    for ds_id, stats in sorted(max_query_lengths.items()):
        max_len = stats["max_len"]
        sample_len = stats["sample_len"]
        sample_query = stats["sample"].replace("\n", " ")
        if len(sample_query) > 300:
            sample_query = sample_query[:297] + "..."

        print(f"{ds_id:<50} | {max_len:<5} | {sample_len:<5} | {sample_query}")
    print("="*120 + "\n")
