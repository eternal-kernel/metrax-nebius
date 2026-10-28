"""Automated verification test for Tab 2 (URL audit) and Tab 3 (Image OCR inspection)."""

import base64
import os
from engine import (
    extract_text_from_image_base64,
    fetch_url_or_regulatory_context,
    run_full_inspection,
)

def test_tab_2_url_ingestion():
    print("=" * 70)
    print("TESTING TAB 2: LIVE URL AUDIT VIA TAVILY (WITH SEARCH FALLBACK)")
    print("=" * 70)

    test_urls = [
        "https://world.openfoodfacts.org/product/8901725181222/aashirvaad-superior-mp-atta",
        "https://pib.gov.in/PressReleasePage.aspx?PRID=1785536"
    ]

    for url in test_urls:
        print(f"\n--> Testing URL Ingestion: {url}")
        content, citations, tel = fetch_url_or_regulatory_context(url, is_url=True)
        print(f"    Tool Used       : {tel.model_or_tool}")
        print(f"    Reason          : {tel.reason}")
        print(f"    Latency         : {tel.latency_ms} ms")
        print(f"    Extracted Length: {len(content)} characters")
        print(f"    Citations Count : {len(citations)}")
        assert len(content) > 50, f"Expected content from {url}, got empty"
        print(f"    [SUCCESS] Ingested successfully!")

    print("\n--> Running Full Inspection on Ingested URL (Aashirvaad Atta)...")
    url = test_urls[0]
    report = run_full_inspection(url, is_url=True)
    print(f"    Compliance Score: {report.deterministic_audit.compliance_score}/100")
    print(f"    Status          : {report.deterministic_audit.status}")
    print(f"    Product Name    : {report.extracted_declarations.product_name}")
    print(f"    Cost Savings    : {report.estimated_cost_savings_pct}%")
    print(f"\n[SUCCESS] Tab 2 URL Ingestion & Audit Verified Flawlessly!\n")


def test_tab_3_label_ocr_inspection():
    print("=" * 70)
    print("TESTING TAB 3: PACKAGING LABEL PHOTO OCR VIA NEMOTRON-3.5-SUPER-VL")
    print("=" * 70)

    sample_img_path = "sample_label.png"
    assert os.path.exists(sample_img_path), f"File {sample_img_path} not found"

    print(f"\n--> Reading {sample_img_path} and encoding to base64...")
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()
    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    print("--> Transcribing via NVIDIA Nemotron-3.5-Super-VL...")
    transcribed_text, vl_telemetry = extract_text_from_image_base64(b64_img, mime_type="image/png")
    print(f"    Model Used      : {vl_telemetry.model_or_tool}")
    print(f"    Latency         : {vl_telemetry.latency_ms} ms")
    print(f"    Prompt Tokens   : {vl_telemetry.prompt_tokens}")
    print(f"    Completion Toks : {vl_telemetry.completion_tokens}")
    print(f"    Est. Cost       : ${vl_telemetry.estimated_cost_usd:.5f}")
    print(f"\nTranscribed Preview:\n{transcribed_text[:350]}...\n")

    assert len(transcribed_text) > 50, "OCR transcription was empty"
    assert "TITAN" in transcribed_text.upper() or "WHEY" in transcribed_text.upper(), "Expected product name in OCR text"

    print("--> Routing Transcribed Text to Full Inspection Pipeline...")
    report = run_full_inspection(transcribed_text, is_url=False, force_ultra=False)

    print(f"    Extracted Product Name : {report.extracted_declarations.product_name}")
    print(f"    Net Quantity Value/Unit: {report.extracted_declarations.net_quantity_value} {report.extracted_declarations.net_quantity_unit}")
    print(f"    Compliance Score       : {report.deterministic_audit.compliance_score}/100")
    print(f"    Status                 : {report.deterministic_audit.status}")
    print(f"    Defects Caught         : {len(report.deterministic_audit.violations)}")
    for idx, v in enumerate(report.deterministic_audit.violations, 1):
        print(f"      #{idx} [{v.severity}] {v.rule_citation} ({v.field_name})")

    assert report.deterministic_audit.compliance_score < 70, "Expected non-compliant score for sample label"
    assert len(report.deterministic_audit.violations) >= 3, "Expected multiple statutory violations caught"
    assert report.ultra_legal_memo is not None and len(report.ultra_legal_memo) > 200, "Expected legal memo generated"

    print(f"\n[SUCCESS] Tab 3 Image OCR & Multi-Tier Inspection Verified Flawlessly!\n")


if __name__ == "__main__":
    test_tab_2_url_ingestion()
    test_tab_3_label_ocr_inspection()
