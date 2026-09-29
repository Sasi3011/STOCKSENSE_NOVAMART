// Builds presentation/StockSense_Pitch.pptx from pipeline outputs (/tmp/deck.json)
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const D = JSON.parse(fs.readFileSync(process.argv[2] || "/tmp/deck.json", "utf8"));
const OUT = process.argv[3] || "StockSense_Pitch.pptx";

const INK = "1B2A2F", PAPER = "F4F6F3", TAG = "FFD23F", TEAL = "1F5F6B", SAGE = "8DB3A8",
      MUTED = "5E6F74", RULE = "D5DBD8", HIGH = "C8332F", MED = "D9822B", LOW = "2E7D5B", WHITE = "FFFFFF";
const F = "Arial";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "StockSense - NovaMart";

const rs = v => { v = +v; if (Math.abs(v) >= 1e7) return "Rs " + (v / 1e7).toFixed(2) + " Cr";
  if (Math.abs(v) >= 1e5) return "Rs " + (v / 1e5).toFixed(1) + " L"; if (Math.abs(v) >= 1e3) return "Rs " + (v / 1e3).toFixed(1) + " K"; return "Rs " + Math.round(v); };
const pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
const n = v => Math.round(v).toLocaleString("en-IN");

function base(title, sub, dark = false) {
  const s = pres.addSlide();
  s.background = { color: dark ? INK : PAPER };
  if (title) s.addText(title, { x: 0.6, y: 0.4, w: 12.1, h: 0.7, fontFace: F, fontSize: 28, bold: true, color: dark ? PAPER : INK, margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x: 0.6, y: 1.1, w: 12.1, h: 0.45, fontFace: F, fontSize: 14, color: dark ? "B8C7C9" : MUTED, margin: 0, isTextBox: true });
  return s;
}
// the shelf-edge tag motif
function tag(s, x, y, w, h, big, label, risk = null, bigSize = 34) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: TAG }, line: { color: TAG } });
  if (risk) s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.09, h, fill: { color: risk }, line: { color: risk } });
  s.addShape(pres.shapes.OVAL, { x: x + w - 0.3, y: y + 0.16, w: 0.13, h: 0.13, fill: { color: PAPER }, line: { color: "C9A92E", width: 0.5 } });
  s.addText(big, { x: x + 0.25, y: y + 0.18, w: w - 0.6, h: h * 0.52, fontFace: F, fontSize: bigSize, bold: true, color: INK, margin: 0, valign: "bottom", isTextBox: true });
  s.addText(label, { x: x + 0.25, y: y + h * 0.62, w: w - 0.4, h: h * 0.33, fontFace: F, fontSize: 12, color: INK, margin: 0, valign: "top", isTextBox: true });
}
function note(s, text, x, y, w, h, dark = false) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h, fill: { color: TEAL }, line: { color: TEAL } });
  s.addText(text, { x: x + 0.2, y, w: w - 0.2, h, fontFace: F, fontSize: 13, color: dark ? PAPER : INK, margin: 0, valign: "middle", isTextBox: true });
}
const chartBase = (extra) => Object.assign({ catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: F, valAxisLabelFontFace: F,
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, valGridLine: { color: RULE, size: 0.5 }, catGridLine: { style: "none" },
  showLegend: false, dataLabelFontFace: F, dataLabelFontSize: 10, dataLabelColor: INK, titleFontFace: F, titleFontSize: 13, titleColor: INK }, extra);

const k = D.kpi, reg = D.reg, clf = D.clf;
const regTest = Object.fromEntries(reg.test.map(x => [x.model, x]));
const sel = regTest[reg.selected], baseM = regTest["Baseline (7-day mean x 7)"], ct = clf.test[0];

// 1. Title -------------------------------------------------------------------
let s = base(null, null, true);
s.addText("StockSense", { x: 0.8, y: 1.9, w: 8, h: 1.3, fontFace: F, fontSize: 66, bold: true, color: PAPER, margin: 0, isTextBox: true });
s.addText("Predict demand. Prevent stock-outs. Power better decisions.", { x: 0.8, y: 3.2, w: 8.5, h: 0.6, fontFace: F, fontSize: 20, color: "B8C7C9", margin: 0, isTextBox: true });
s.addText("NovaMart Retail Pvt. Ltd.  |  IntelliData 2026  |  Team: [your team name]", { x: 0.8, y: 6.3, w: 9, h: 0.4, fontFace: F, fontSize: 13, color: "8FA3A6", margin: 0, isTextBox: true });
tag(s, 9.4, 2.0, 3.2, 1.9, `${D.risk.HIGH || 0}`, "store-product lines need an order today", HIGH, 54);
s.addNotes("Open with the challenge line: every empty shelf is lost revenue. We built a working decision tool, not just a model.");

// 2. Problem -----------------------------------------------------------------
s = base("Every empty shelf is lost revenue", `Five months of NovaMart data, six stores, ${k.date_min} to ${k.date_max}`);
tag(s, 0.6, 1.9, 3.8, 1.8, rs(k.estimated_lost_sales), "estimated sales lost to empty shelves", HIGH);
tag(s, 4.75, 1.9, 3.8, 1.8, pct(k.stockout_rate, 1), "of store-product days end with an empty shelf", MED);
tag(s, 8.9, 1.9, 3.8, 1.8, rs(k.revenue_total), "total revenue in the period", LOW);
const gr = Object.entries(D.growth).sort((a, b) => b[1] - a[1]);
s.addChart(pres.charts.BAR, [{ name: "Growth", labels: gr.map(x => x[0]), values: gr.map(x => +(x[1] * 100).toFixed(1)) }],
  chartBase({ x: 0.6, y: 4.0, w: 6, h: 3.1, barDir: "bar", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0\"%\"",
    showTitle: true, title: "Store growth, first 4 vs last 4 weeks (%)" }));
s.addChart(pres.charts.BAR, [{ name: "Stock-out", labels: gr.map(x => x[0]), values: gr.map(x => +(D.so[x[0]] * 100).toFixed(1)) }],
  chartBase({ x: 6.8, y: 4.0, w: 6, h: 3.1, barDir: "bar", chartColors: [MED], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"",
    showTitle: true, title: "Stock-out rate by store (%)" }));
s.addNotes(`Headline: ${gr[0][0]} grew fastest and has the highest stock-out rate. Fixed reorder levels do not scale with growth.`);

// 3. Data --------------------------------------------------------------------
s = base("Messy data in, one trusted table out", "Five raw files merged to one row per date x store x product");
const flow = ["transactions", "products", "stores", "inventory", "external factors"];
flow.forEach((f, i) => { s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.9 + i * 0.62, w: 2.6, h: 0.5, fill: { color: WHITE }, line: { color: RULE } });
  s.addText(f, { x: 0.6, y: 1.9 + i * 0.62, w: 2.6, h: 0.5, fontFace: F, fontSize: 13, color: INK, align: "center", valign: "middle", margin: 0, isTextBox: true }); });
s.addShape(pres.shapes.RIGHT_ARROW, { x: 3.4, y: 3.1, w: 0.7, h: 0.5, fill: { color: TEAL }, line: { color: TEAL } });
tag(s, 4.3, 2.3, 3.2, 2.1, "41,370", "rows in the master table, zero missing values", LOW, 40);
const rows = [[{ text: "Issue", options: { bold: true, fill: { color: "E8ECE9" } } }, { text: "Count", options: { bold: true, fill: { color: "E8ECE9" }, align: "right" } }]]
  .concat(D.dq_top.map(r => [`${r[1]} (${r[0]})`, { text: n(r[2]), options: { align: "right" } }]));
s.addTable(rows, { x: 7.9, y: 1.9, w: 4.8, colW: [3.8, 1.0], fontFace: F, fontSize: 11, color: INK, border: { type: "solid", color: RULE, pt: 0.5 }, rowH: 0.42 });
note(s, `${D.dq_issues} issue types, ${n(D.dq_rows)} records fixed. Every one is logged with the action taken and why (reports/data_quality_report.md). New products with 5 days of history use a look-alike fallback.`, 0.6, 5.3, 12.1, 1.0);

// 4. Discoveries -------------------------------------------------------------
s = base("What the data told us", "Four findings that shaped the models, each confirmed by a hypothesis test");
const finds = [[`+${pct(D.promo)}`, "units on promotion days"], [`+${pct(D.weekend)}`, "units on weekends"],
  [`${D.pareto}`, "products make 80% of revenue"], [`${gr[0][0]} +${pct(gr[0][1])}`, "growth, yet the emptiest shelves"]];
finds.forEach((f, i) => tag(s, 0.6 + i * 3.1, 1.85, 2.85, 1.6, f[0], f[1], null, 28));
const st = [[{ text: "Question", options: { bold: true, fill: { color: "E8ECE9" } } }, { text: "Test", options: { bold: true, fill: { color: "E8ECE9" } } },
  { text: "p-value", options: { bold: true, fill: { color: "E8ECE9" }, align: "right" } }, { text: "Effect", options: { bold: true, fill: { color: "E8ECE9" } } }]]
  .concat(D.stats.map(t => [t.q, t.t, { text: t.p.toExponential(1), options: { align: "right" } }, t.e]));
s.addTable(st, { x: 0.6, y: 3.8, w: 12.1, colW: [3.6, 2.9, 1.1, 4.5], fontFace: F, fontSize: 10.5, color: INK, border: { type: "solid", color: RULE, pt: 0.5 }, rowH: 0.52 });
s.addNotes("All four tests reject H0 at alpha 0.05. The promotion-stock-out association is real but small on its own - it matters in combination with low stock, which the model captures.");

// 5. Design against leakage --------------------------------------------------
s = base("Built so the scores can be trusted", "Time-ordered, purged splits: the model never sees the future");
const seg = [["Train", D.splits.train, TEAL, 6.2], ["gap", null, RULE, 0.5], ["Validate", D.splits.valid, SAGE, 1.6], ["gap", null, RULE, 0.5], ["Test", D.splits.test, TAG, 2.6]];
let x = 0.6;
seg.forEach(([l, sp, c, w]) => { s.addShape(pres.shapes.RECTANGLE, { x, y: 2.0, w, h: 0.8, fill: { color: c }, line: { color: c } });
  s.addText(sp ? `${l}\n${sp[0]} to ${sp[1]}` : "7 days", { x, y: 2.0, w, h: 0.8, fontFace: F, fontSize: sp ? 12 : 9, color: l === "Train" ? WHITE : INK, align: "center", valign: "middle", margin: 0, isTextBox: true }); x += w + 0.05; });
const pts = [["49 features, 6 groups", "Time, lags and rolling windows, inventory cover, price and promotion, weather, store and product."],
  ["Only what is known at prediction time", "Rolling windows end today; targets start tomorrow. The festival calendar is the only forward input, because it is published in advance."],
  ["7-day purge between splits", "Stops training targets overlapping validation days, a subtle leak that inflates scores."],
  ["Permitted algorithms only", "Linear Regression, Decision Tree, Random Forest, XGBoost, KNN, SVM, Naive Bayes."]];
pts.forEach((p, i) => { const yy = 3.3 + i * 0.95;
  s.addText(p[0], { x: 0.6, y: yy, w: 4.2, h: 0.8, fontFace: F, fontSize: 15, bold: true, color: INK, margin: 0, valign: "top", isTextBox: true });
  s.addText(p[1], { x: 4.9, y: yy, w: 7.8, h: 0.8, fontFace: F, fontSize: 13, color: MUTED, margin: 0, valign: "top", isTextBox: true }); });

// 6. Demand model ------------------------------------------------------------
s = base("Model 1: 7-day demand forecast", `Selected: ${reg.selected}, chosen on validation WAPE`);
const rv = reg.validation.slice().sort((a, b) => a.WAPE - b.WAPE);
s.addChart(pres.charts.BAR, [{ name: "WAPE", labels: rv.map(x => x.model.replace(" (7-day mean x 7)", "")), values: rv.map(x => +x.WAPE.toFixed(1)) }],
  chartBase({ x: 0.6, y: 1.8, w: 7.2, h: 5.0, barDir: "bar", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"",
    showTitle: true, title: "Validation error by model (WAPE %, lower is better)", catAxisOrientation: "maxMin" }));
tag(s, 8.3, 1.9, 4.4, 1.5, `${sel.WAPE.toFixed(1)}% WAPE`, `on the untouched test weeks, vs ${baseM.WAPE.toFixed(1)}% for the old 7-day-average rule`, LOW, 30);
tag(s, 8.3, 3.6, 4.4, 1.5, `R² ${sel.R2.toFixed(3)}`, `MAE ${sel.MAE.toFixed(1)} units per item per week`, null, 30);
note(s, "WAPE is the business metric: orders are placed in units, and MAPE blows up on slow movers.", 8.3, 5.4, 4.4, 1.2);

// 7. Risk model --------------------------------------------------------------
s = base("Model 2: stock-out risk in the next 7 days", `Selected: ${clf.selected}, calibrated so a 70% risk really means about 7 in 10`);
const cv = clf.validation.slice().sort((a, b) => b.PR_AUC - a.PR_AUC);
s.addChart(pres.charts.BAR, [{ name: "PR-AUC", labels: cv.map(x => x.model), values: cv.map(x => +x.PR_AUC.toFixed(3)) }],
  chartBase({ x: 0.6, y: 1.8, w: 6.2, h: 4.2, barDir: "bar", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    showTitle: true, title: "Validation PR-AUC by model (higher is better)", catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 0.6 }));
const cm = clf.confusion_matrix;
const cmRows = [["", { text: "Predicted fine", options: { bold: true } }, { text: "Predicted stock-out", options: { bold: true } }],
  [{ text: "Actually fine", options: { bold: true } }, n(cm[0][0]), n(cm[0][1])], [{ text: "Stock-out", options: { bold: true } }, n(cm[1][0]), { text: n(cm[1][1]), options: { bold: true, fill: { color: TAG } } }]];
s.addTable(cmRows, { x: 7.2, y: 1.9, w: 5.5, colW: [1.7, 1.9, 1.9], fontFace: F, fontSize: 13, color: INK, align: "center", valign: "middle", border: { type: "solid", color: RULE, pt: 0.5 }, rowH: 0.55 });
tag(s, 7.2, 3.8, 2.6, 1.4, ct.ROC_AUC.toFixed(3), "ROC-AUC, test", null, 28);
tag(s, 10.1, 3.8, 2.6, 1.4, pct(ct.Recall), "of stock-outs caught", LOW, 28);
note(s, `Imbalance: class weights and scale_pos_weight, ranked by PR-AUC. Alert threshold ${clf.threshold.toFixed(2)} maximises F2 with precision of at least 45%: a missed stock-out costs sales, a false alarm costs a shelf check.`, 0.6, 6.2, 12.1, 0.9);

// 8. Action card -------------------------------------------------------------
const c = D.card, reasons = JSON.parse(c.reasons).map(r => r[1] === "rule" ? r[0] : `${r[0]} (${r[1]})`).join("; ");
s = base("From prediction to action", "Every flag becomes a quantity, a reason and an instruction");
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.8, w: 6.0, h: 5.0, fill: { color: TAG }, line: { color: TAG } });
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.8, w: 0.12, h: 5.0, fill: { color: HIGH }, line: { color: HIGH } });
s.addShape(pres.shapes.OVAL, { x: 6.2, y: 2.0, w: 0.16, h: 0.16, fill: { color: PAPER }, line: { color: "C9A92E", width: 0.5 } });
s.addText([{ text: `STORE ${c.store_id} - ${c.product_name.toUpperCase()}`, options: { bold: true, fontSize: 20, breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: `Predicted 7-day demand: ${n(c.forecast_7d)} units`, options: { breakLine: true } },
  { text: `Current stock: ${n(c.current_stock)} units  |  Incoming: 0`, options: { breakLine: true } },
  { text: `Stock-out probability: ${pct(c.stockout_probability)}  |  Risk: ${c.risk_level}`, options: { breakLine: true } },
  { text: `Shelf life ${c.shelf_life_days} days, so stock is capped at ${n(c.recommended_stock)}`, options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: `Order ${n(c.reorder_qty)} units`, options: { bold: true, fontSize: 30, breakLine: true } },
  { text: `Why? ${reasons}`, options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: `MANAGER ACTION: ${c.manager_action}`, options: { bold: true } }],
  { x: 0.95, y: 2.0, w: 5.4, h: 4.6, fontFace: F, fontSize: 14, color: INK, margin: 0, valign: "top", isTextBox: true });
const logic = [["Risk band", "High if p >= 0.70; Medium if 0.40 to 0.70"], ["Safety stock", "z x daily std x sqrt(lead time); z = 1.65 / 1.28 / 1.00"],
  ["Recommended stock", "7-day forecast + safety stock, capped at shelf life"], ["Reorder quantity", "max(0, recommended - current - incoming), in case packs"],
  ["Guard-rails", "Empty shelf, or stock runs out before a new order lands, means High. Always labelled as a rule."],
  ["Reasons", "Each driver is reset to normal; the change in prediction is its impact"]];
logic.forEach((l, i) => { const yy = 1.85 + i * 0.83;
  s.addText(l[0], { x: 7.0, y: yy, w: 5.7, h: 0.3, fontFace: F, fontSize: 14, bold: true, color: INK, margin: 0, isTextBox: true });
  s.addText(l[1], { x: 7.0, y: yy + 0.3, w: 5.7, h: 0.45, fontFace: F, fontSize: 12, color: MUTED, margin: 0, isTextBox: true }); });

// 9. Today -------------------------------------------------------------------
s = base("Who needs action right now", `As of ${D.asof}, forecast for the next 7 days`);
tag(s, 0.6, 1.8, 2.9, 1.6, `${D.risk.HIGH || 0}`, "high risk: order today", HIGH, 40);
tag(s, 3.7, 1.8, 2.9, 1.6, `${D.risk.MEDIUM || 0}`, "medium: order within 2 days", MED, 40);
tag(s, 6.8, 1.8, 2.9, 1.6, n(D.units), `units across ${D.lines} order lines`, null, 36);
tag(s, 9.9, 1.8, 2.8, 1.6, rs(D.exv), `tied up in ${D.nover} overstocked items`, LOW, 30);
const tr = [["Store", "Product", "Stock", "7-day forecast", "Risk", "Order"].map(h => ({ text: h, options: { bold: true, fill: { color: "E8ECE9" } } }))]
  .concat(D.top.map(r => [r[0], r[1], { text: n(r[2]), options: { align: "right" } }, { text: n(r[3]), options: { align: "right" } }, r[4], { text: n(r[5]), options: { align: "right", bold: true } }]));
s.addTable(tr, { x: 0.6, y: 3.7, w: 12.1, colW: [1.2, 4.3, 1.4, 2.0, 1.6, 1.6], fontFace: F, fontSize: 12, color: INK, border: { type: "solid", color: RULE, pt: 0.5 }, rowH: 0.45 });

// 10. What-if ----------------------------------------------------------------
s = base("What-if: planning before the problem", "Both models re-run under each scenario");
s.addChart(pres.charts.BAR, [{ name: "High-risk items", labels: D.whatif.map(x => x[0]), values: D.whatif.map(x => x[1]) }],
  chartBase({ x: 0.6, y: 1.8, w: 7.4, h: 5.0, barDir: "col", chartColors: [HIGH], showValue: true, dataLabelPosition: "outEnd", showTitle: true, title: "High-risk items under each scenario" }));
note(s, `A 2-day supplier delay is the biggest threat: high-risk items rise from ${D.whatif[0][1]} to ${D.whatif[4][1]}. Promotions and festivals raise demand, and the order plan grows with them, so the promotion does not create its own stock-out.`, 8.4, 2.0, 4.3, 2.4);
note(s, "Supplier delays change risk only, never demand. Promotions are modelled as a real price cut against the item's recent price.", 8.4, 4.8, 4.3, 1.6);

// 11. Close ------------------------------------------------------------------
s = base("How this reduces business loss", null, true);
const close = [["Fewer empty shelves", `Stock-outs flagged up to 7 days ahead; ${pct(ct.Recall)} caught on the test weeks, with a quantity and a reason attached.`],
  ["Less cash on the shelf", `${D.nover} overstocked items flagged to hold; perishables capped by shelf life to cut expiry.`],
  ["Explainable and repeatable", "Every alert says why. One command rebuilds everything from new data."],
  ["Honest limits", "Sales are capped by stock (censored demand); festival effects are under-learned from few examples; open purchase orders are not yet in the data. The next step is to connect them."]];
close.forEach((c2, i) => { const yy = 1.5 + i * 1.3;
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: yy + 0.05, w: 0.14, h: 0.9, fill: { color: i < 3 ? TAG : MED }, line: { color: i < 3 ? TAG : MED } });
  s.addText(c2[0], { x: 1.2, y: yy, w: 11, h: 0.4, fontFace: F, fontSize: 18, bold: true, color: PAPER, margin: 0, isTextBox: true });
  s.addText(c2[1], { x: 1.2, y: yy + 0.42, w: 11, h: 0.6, fontFace: F, fontSize: 13.5, color: "B8C7C9", margin: 0, isTextBox: true }); });
s.addText("Every empty shelf is lost revenue. Now we see it coming.", { x: 0.8, y: 6.7, w: 11.5, h: 0.5, fontFace: F, fontSize: 16, bold: true, color: TAG, margin: 0, isTextBox: true });

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote " + f));
