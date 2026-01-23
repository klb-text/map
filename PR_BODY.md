
# PR: Catalog-accelerated Vehicle → (YMMT) → CADS flow (no alias memory)

This PR replaces `app.py` with a version that introduces a deterministic, alias-free lookup using the `AFF Vehicles YMMT` catalog, and a clean YMM/YMMT fallback. Core changes:

- ✨ **Vehicle (catalog → CADS)** quick lookup button
- 🧭 **YMM / YMMT fallback** pickers powered by the same catalog
- 🚚 **Harvest** route: `?harvest=1&source=catalog&vehicle=...&catalog_path=data/AFF%20Vehicles%20YMMT.csv&plain=1`
- ♻️ Reuses existing CADS filtering tiers and GitHub persistence for mappings

No alias memory is used; make/model lists come directly from the catalog. If OEMs add items, update `data/AFF Vehicles YMMT.csv` and the UI will reflect it automatically.

## How to use
1. Set **Vehicle Catalog path** in the sidebar (default `data/AFF Vehicles YMMT.csv`).
2. Paste a **Vehicle** (e.g. `2026 Lexus RX 350 Premium AWD`) and click **⚡ Vehicle (catalog → CADS)**.
3. If ambiguous, use **YMM / YMMT** fallback pickers.
4. Confirm a mapping in **Confirm & Save Mapping**; click **Save mapping to GitHub**.

## Notes
- Mappings write to `data/mappings.json` (same path as before).
- The app honors existing matching controls (trim hint, stopwords, exact year, etc.).

---
