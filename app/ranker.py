"""Candidate ranking module for the Resume Analyzer.

Sorts analyzed candidates by match score with deterministic tie-breaking
and supports Top-N shortlisting.
"""

from __future__ import annotations


def rank_candidates(results: list[dict], top_n: int | None = None) -> dict:
    """Rank analyzed candidates by match score with deterministic tie-breaking.

    Parameters
    ----------
    results : list[dict]
        The ``results`` list from the ``/resume/bulk-analyze`` endpoint.
        Only entries with ``status == "analyzed"`` are ranked.
    top_n : int or None
        If provided, return only the top N candidates.
        Must be >= 1. If greater than the total analyzed count,
        all candidates are returned with a notice.

    Returns
    -------
    dict
        ``ranked``: list of ranked candidate dicts with rank numbers.
        ``total_analyzed``: int — total candidates that were analyzed.
        ``top_n_requested``: int or None — the requested N.
        ``top_n_returned``: int — actual number returned.
        ``notice``: str or None — informational message if N > total.

    Raises
    ------
    ValueError
        If ``top_n`` is less than 1.
    """
    # Filter to only successfully analyzed entries
    analyzed = [r for r in results if r.get("status") == "analyzed"]
    total_analyzed = len(analyzed)

    # Validate top_n
    notice = None
    if top_n is not None:
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        if top_n > total_analyzed:
            notice = (
                f"Requested top {top_n} but only {total_analyzed} candidates "
                f"were analyzed. Showing all."
            )

    # Sort with deterministic tie-breaking:
    # 1. match_score DESC (higher is better)
    # 2. skills_match_count DESC (more matched skills is better)
    # 3. professional_months DESC (then experience_years * 12 as fallback)
    # 4. freelance_months DESC
    # 5. candidate_name ASC (alphabetical for final tie-break)
    analyzed.sort(
        key=lambda r: (
            -(r.get("match_score") or 0),
            -(r.get("skills_match_count") or 0),
            -(r.get("professional_months") if r.get("professional_months") is not None else ((r.get("experience_years") or 0) * 12)),
            -(r.get("freelance_months") or 0),
            (r.get("candidate_name") or r.get("filename", "")).lower(),
        )
    )

    # Slice to top N if requested
    sliced = analyzed[:top_n] if top_n is not None else analyzed

    # Build ranked output with rank numbers and reasons
    ranked = []
    for i, entry in enumerate(sliced, start=1):
        score_pct = int(round((entry.get("match_score") or 0) * 100))
        skills_count = entry.get("skills_match_count") or 0
        total_req = entry.get("total_required") or 0
        exp = entry.get("experience_years")
        prof_str = entry.get("professional_experience_str") or (f"{exp} yrs" if exp else "Not indicated")
        free_str = entry.get("freelance_experience_str") or "Not indicated"
        exp_str = f"Prof: {prof_str}, Freelance: {free_str}" if prof_str != "Not indicated" or free_str != "Not indicated" else "experience not specified"
        req_matched = entry.get("required_matched") or []
        req_missing = entry.get("required_missing") or []
        pref_matched = entry.get("preferred_matched") or []
        pref_missing = entry.get("preferred_missing") or []

        reason = (
            f"Rank #{i}: {score_pct}% match score, "
            f"{len(req_matched)} of {total_req} required skills matched, "
            f"{exp_str}"
        )

        ranked.append({
            "rank": i,
            "filename": entry.get("filename", ""),
            "candidate_name": entry.get("candidate_name", ""),
            "match_score": entry.get("match_score", 0),
            "required_matched": req_matched,
            "required_missing": req_missing,
            "preferred_matched": pref_matched,
            "preferred_missing": pref_missing,
            "total_required": total_req,
            "total_preferred": entry.get("total_preferred") or 0,
            "skills_match_count": skills_count,
            "experience_years": exp,
            "professional_months": entry.get("professional_months", 0),
            "freelance_months": entry.get("freelance_months", 0),
            "professional_experience_str": prof_str,
            "freelance_experience_str": free_str,
            "combined_experience_str": entry.get("combined_experience_str"),
            "internship_note": entry.get("internship_note"),
            "education": entry.get("education") or [],
            "rank_reason": reason,
        })

    return {
        "ranked": ranked,
        "total_analyzed": total_analyzed,
        "top_n_requested": top_n,
        "top_n_returned": len(ranked),
        "notice": notice,
    }
