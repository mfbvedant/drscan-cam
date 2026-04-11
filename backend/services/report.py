"""
DRScan Cam — Clinical Report Builder
Generates structured clinical findings and recommendations per modality.
"""

from typing import List
from backend.config import Modality, SEVERITY_LEVELS, MODALITY_INFO


def generate_report(
    modality: str,
    diagnosis: str,
    severity: int,
    confidence: float,
    regions: List[str],
    top_predictions: list,
) -> dict:
    """
    Generate a structured clinical report from prediction results.

    Returns dict with:
      - summary: one-line summary
      - findings: detailed findings text
      - recommendation: clinical recommendation
      - disclaimer: legal/medical disclaimer
    """
    sev = SEVERITY_LEVELS.get(severity, SEVERITY_LEVELS[0])
    mod_info = MODALITY_INFO.get(modality, {})
    mod_name = mod_info.get("name", modality)

    # --- Summary ---
    summary = f"{mod_name} Analysis: {diagnosis} — {sev['label']} (Confidence: {confidence:.1%})"

    # --- Findings ---
    findings_lines = [
        f"**Modality:** {mod_name}",
        f"**Primary Finding:** {diagnosis}",
        f"**Severity Grade:** {severity}/4 ({sev['label']})",
        f"**Confidence Score:** {confidence:.1%}",
        "",
        "**Regions of Interest:**",
    ]
    for region in regions:
        findings_lines.append(f"  • {region}")

    if len(top_predictions) > 1:
        findings_lines.append("")
        findings_lines.append("**Differential Considerations:**")
        for pred in top_predictions[1:4]:  # Top 3 alternatives
            findings_lines.append(
                f"  • {pred['class_name']}: {pred['confidence']:.1%}"
            )

    findings = "\n".join(findings_lines)

    # --- Modality-specific observations ---
    observations = _get_modality_observations(modality, diagnosis, severity)
    if observations:
        findings += "\n\n**Additional Observations:**\n" + observations

    # --- Recommendation ---
    recommendation = sev["recommendation"]

    # --- Disclaimer ---
    disclaimer = (
        "⚠️ This AI-generated report is intended as a clinical decision support tool only. "
        "It does not constitute a medical diagnosis. All findings must be reviewed and "
        "confirmed by a qualified healthcare professional. AI predictions should be "
        "correlated with clinical history, physical examination, and other diagnostic tests."
    )

    return {
        "summary": summary,
        "findings": findings,
        "recommendation": recommendation,
        "disclaimer": disclaimer,
    }


def _get_modality_observations(modality: str, diagnosis: str, severity: int) -> str:
    """Generate modality-specific clinical observations."""

    if modality == Modality.CHEST_XRAY:
        obs = {
            "Normal": "Heart size and mediastinal silhouette appear within normal limits. Lungs are clear bilaterally. No pleural effusion or pneumothorax identified.",
            "Bacterial Pneumonia": "Focal consolidation identified. Air bronchograms may be present. Clinical correlation with symptoms, lab values (WBC, CRP), and patient history recommended.",
            "Viral Pneumonia": "Bilateral interstitial infiltrates noted. Pattern consistent with viral pneumonitis. Consider COVID-19 testing if clinically indicated.",
            "Pleural Effusion": "Blunting of costophrenic angle detected. Consider lateral decubitus view for confirmation. Thoracentesis may be indicated for large effusions.",
            "Cardiomegaly": "Cardiothoracic ratio appears increased. Consider echocardiography for functional assessment. Evaluate for signs of heart failure.",
            "Pneumothorax": "Visceral pleural line visualized. Assess for tension features (mediastinal shift, diaphragm flattening). Urgent clinical assessment required.",
        }
    elif modality == Modality.BRAIN_MRI:
        obs = {
            "Normal": "No intracranial mass lesion identified. Ventricles and sulci are age-appropriate. No midline shift or mass effect.",
            "Glioma": "Intra-axial mass lesion identified with heterogeneous signal characteristics. Further characterization with contrast-enhanced MRI recommended. Neurosurgical consultation advised.",
            "Meningioma": "Extra-axial mass with broad dural base identified. Homogeneous enhancement pattern. Typically benign — monitor for growth or symptoms.",
            "Pituitary Tumor": "Sellar/suprasellar mass identified. Evaluate visual fields. Endocrine workup recommended (prolactin, GH, cortisol, thyroid function).",
            "Metastatic Lesion": "Multiple enhancing lesions at gray-white junction. Pattern suggestive of metastatic disease. Primary malignancy workup recommended.",
        }
    elif modality == Modality.LUNG_CT:
        obs = {
            "Normal": "No pulmonary nodules or masses identified. Airways are patent. No pleural disease.",
            "Lung Nodule": "Pulmonary nodule identified. Size, morphology, and location should be documented. Follow Fleischner Society guidelines for management.",
            "Ground Glass Opacity": "Ground glass opacity detected. Differential includes infection, inflammation, early malignancy, or hemorrhage. Clinical correlation essential.",
            "Emphysema": "Centrilobular or panlobular emphysematous changes noted. Quantitative assessment may be helpful. Pulmonary function testing recommended.",
            "Pulmonary Fibrosis": "Reticular pattern with possible honeycombing identified. Consider UIP vs NSIP pattern. Multidisciplinary discussion recommended.",
        }
    else:
        return ""

    return obs.get(diagnosis, "Findings documented. Clinical correlation recommended.")
