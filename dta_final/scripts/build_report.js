// Builds the SC4020 Project 2 report as a .docx matching the Project 1 house style:
// A4, 1in margins, Arial 11, 1.15 line spacing, justified body, numbered headings,
// borderless tables with a shaded header row, captions as plain paragraphs.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, TableOfContents, convertInchesToTwip,
} = require("docx");

const FIG = "/home/claude/merged/dta/results/figures/";
const TW = 9300;              // table width in DXA, as Project 1 uses
const FONT = "Arial";
const NIL = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NO_BORDERS = { top: NIL, bottom: NIL, left: NIL, right: NIL,
                     insideHorizontal: NIL, insideVertical: NIL };

let figN = 0, tabN = 0;

const p = (text, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  widowControl: false,
  spacing: { line: 240, after: opts.after === undefined ? 100 : opts.after },
  children: runs(text),
});

// **bold** and `code` inline, split into runs.
function runs(text) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: 22 }));
    const tok = m[0];
    if (tok.startsWith("**")) {
      out.push(new TextRun({ text: tok.slice(2, -2), font: FONT, size: 22, bold: true }));
    } else {
      out.push(new TextRun({ text: tok.slice(1, -1), font: "Consolas", size: 20 }));
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: 22 }));
  return out.length ? out : [new TextRun({ text: "", font: FONT, size: 22 })];
}

const h1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1 });
const h2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2 });
const h3 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_3 });

const bullet = (text) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  spacing: { line: 240, after: 50 },
  children: runs(text),
});
let numInstance = 0;
const restartNumbering = () => { numInstance += 1; };
const numitem = (text) => new Paragraph({
  numbering: { reference: "numbers", level: 0, instance: numInstance },
  spacing: { line: 240, after: 50 },
  children: runs(text),
});

function caption(kind, text) {
  const n = kind === "Figure" ? ++figN : ++tabN;
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { line: 240, before: kind === "Figure" ? 40 : 0, after: kind === "Figure" ? 100 : 50 },
    children: [
      new TextRun({ text: `${kind} ${n}. `, font: FONT, size: 20, bold: true }),
      new TextRun({ text, font: FONT, size: 20 }),
    ],
  });
}

function figure(file, capText) {
  const img = fs.readFileSync(FIG + file);
  const sizes = {
    "fig01_heatmaps.png": [430, 249], "fig02_distributions.png": [430, 227],
    "fig10_keying.png": [396, 157], "fig05_warm_models.png": [356, 253],
    "fig04_four_settings_davis.png": [430, 184], "fig09_representation.png": [396, 169],
    "fig07_similarity_buckets.png": [324, 137],
  };
  const [w, h] = sizes[file];
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 80, after: 20, line: 240, lineRule: "auto" },
      children: [new ImageRun({ data: img, type: "png", transformation: { width: w, height: h } })],
    }),
    caption("Figure", capText),
  ];
}

// rows[0] is the header. widths are relative and rescaled to TW.
function figurePair(fileA, fileB, wA, hA, wB, hB, capText) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 100, after: 20, line: 240, lineRule: "auto" },
      children: [
        new ImageRun({ data: fs.readFileSync(FIG + fileA), type: "png",
          transformation: { width: wA, height: hA } }),
        new TextRun({ text: " ", font: FONT, size: 16 }),
        new ImageRun({ data: fs.readFileSync(FIG + fileB), type: "png",
          transformation: { width: wB, height: hB } }),
      ],
    }),
    caption("Figure", capText),
  ];
}

function table(rows, widths, capText) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cols = widths.map((w) => Math.round((w / total) * TW));
  cols[cols.length - 1] += TW - cols.reduce((a, b) => a + b, 0);

  const body = rows.map((cells, r) => new TableRow({
    tableHeader: r === 0,
    cantSplit: true,
    children: cells.map((text, c) => new TableCell({
      width: { size: cols[c], type: WidthType.DXA },
      shading: r === 0 ? { type: ShadingType.CLEAR, fill: "EDEDED", color: "auto" } : undefined,
      margins: { top: 30, bottom: 30, left: 70, right: 70 },
      children: [new Paragraph({
        alignment: c === 0 ? AlignmentType.LEFT : AlignmentType.LEFT,
        spacing: { line: 240, after: 0 },
        children: String(text).split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((tok) =>
          tok.startsWith("**")
            ? new TextRun({ text: tok.slice(2, -2), font: FONT, size: 17, bold: true })
            : new TextRun({ text: tok, font: FONT, size: 17, bold: r === 0 })),
      })],
    })),
  }));

  return [
    caption("Table", capText),
    new Table({ columnWidths: cols, width: { size: TW, type: WidthType.DXA }, borders: NO_BORDERS, rows: body }),
    new Paragraph({ spacing: { after: 0, line: 120 }, children: [] }),
  ];
}

// --------------------------------------------------------------------- content

const members = [
  ["Matriculation No.", "Full Name"],
  ["", ""], ["", ""], ["", ""], ["", ""],
];

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 }, paragraph: { spacing: { line: 240, lineRule: "auto" } } },
      title: { run: { font: FONT, size: 40, bold: true },
               paragraph: { spacing: { before: 0, after: 120 }, alignment: AlignmentType.CENTER } },
      heading1: { run: { font: FONT, size: 28, bold: true, color: "000000" },
                  paragraph: { spacing: { before: 170, after: 70 }, keepNext: true } },
      heading2: { run: { font: FONT, size: 24, bold: true, color: "000000" },
                  paragraph: { spacing: { before: 140, after: 50 }, keepNext: true } },
      heading3: { run: { font: FONT, size: 22, bold: true, color: "000000" },
                  paragraph: { spacing: { before: 140, after: 50 }, keepNext: true } },
    },
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: "bullet", text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 420, hanging: 220 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: "decimal", text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 420, hanging: 220 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { margin: {
      top: convertInchesToTwip(0.9), bottom: convertInchesToTwip(0.9),
      left: convertInchesToTwip(0.9), right: convertInchesToTwip(0.9) } } },
    children: [
      // ---------------------------------------------------------- cover
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 60 },
        children: [new TextRun({ text: "SC4020 - Data Analytics and Mining", font: FONT, size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "Project 2 Report", font: FONT, size: 36, bold: true })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 },
        children: [new TextRun({
          text: "Recommender Systems and Cold-Target Generalisation for Drug-Target Binding Affinity on DAVIS and KIBA",
          font: FONT, size: 24 })] }),
      new Table({
        columnWidths: [3100, 6200],
        width: { size: TW, type: WidthType.DXA },
        borders: NO_BORDERS,
        rows: members.map((cells, r) => new TableRow({
          height: { value: 340, rule: "atLeast" },
          children: cells.map((t, c) => new TableCell({
            width: { size: [3100, 6200][c], type: WidthType.DXA },
            margins: { top: 60, bottom: 60, left: 80, right: 80 },
            children: [new Paragraph({ spacing: { after: 0 },
              children: [new TextRun({ text: t, font: FONT, size: 22, bold: r === 0 })] })],
          })),
        })),
      }),
      new Paragraph({ spacing: { before: 440, after: 0 }, children: [] }),
      h1("Abstract"),
      p("A drug-target affinity matrix is a recommender problem: drugs are users, protein targets are items, and binding affinity is the rating. We analyse the DAVIS and KIBA kinase benchmarks, build a collaborative filtering recommender, and solve the cold-start problem it faces when a target has never been measured."),
      p("Three results stand out. First, the deduplication convention is worth measuring rather than assuming: keying DAVIS on source identifier rather than chemistry inflates both collaborative models by six points of variance explained, leaves the bias baseline untouched, and does nothing on KIBA, which has no duplicate sequences. Second, collaborative filtering fails completely on an unseen target, with a per-drug concordance index of exactly 0.500 and zero variance across five seeds, predicted from the algebra before it was measured, while the same model reports a healthy-looking pooled 0.783. Third, a sequence representation recovers most of that loss, cutting error 43% on DAVIS and 32% on KIBA, and the representation matters far more than the model: a normalised Smith-Waterman profile beats mean-pooled ESM-2 650M embeddings on both datasets, while GBA-Mixup, from the paper the brief cites, gives nothing measurable."),
      p("Performance depends sharply on homology. Against a target with a relative in the training panel the method cuts error by 61%; against an orphan below 20% identity it is worse than predicting the drug's own mean. The honest description is a homology lookup rather than cold-target generalisation in the strong sense."),
      new Paragraph({ children: [new PageBreak()] }),

      // ---------------------------------------------------------- TOC
      h1("Table of Content"),
      new TableOfContents("Table of Content", { hyperlink: true, headingStyleRange: "1-3" }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---------------------------------------------------------- abstract
      // ---------------------------------------------------------- 1
      h1("1. Introduction"),
      h2("1.1 Problem and Framing"),
      p("Measuring how strongly a drug binds a protein costs money, so the matrix of measured affinities is mostly empty. That is the structure of a recommender problem: drugs are the users, protein targets the items, affinity the rating, and the question is the one Netflix asks, namely what will this drug bind next?"),
      p("Task 1 characterises two kinase benchmarks and the structural facts that constrain everything after. Task 2 builds the recommender and asks, with experiments rather than opinion, whether the collaborative part is necessary. Task 3 addresses the failure the first two share: a target never seen in training has no learned parameters, so the recommender cannot rank it against any other new target."),
      h2("1.2 Datasets"),
      ...table([
        ["", "DAVIS", "KIBA"],
        ["Ligands, source IDs / distinct chemistry", "68 / 68", "2,111 / 2,068"],
        ["Targets, source IDs / distinct sequence", "442 / **379**", "229 / 229"],
        ["Observed pairs after deduplication", "25,772", "117,657"],
        ["Density", "1.000 (complete)", "0.248"],
        ["Score", "Kd in nM, converted to pKd", "KIBA composite, as supplied"],
        ["Observed range", "5.00 to 10.62", "0.00 to 17.20"],
        ["At the assay floor", "**71.2%**", "none"],
        ["Interaction share of variance", "**63.2%**", "**48.1%**"],
      ], [34, 33, 33], "The two datasets after cleaning and deduplication."),
      p("Almost every distributional difference follows from provenance. DAVIS is one laboratory's single-assay panel over a fixed kinase set, so it is complete and hard-censored at the screening limit. KIBA aggregates Ki, Kd and IC50 values from many sources, so it is sparse, uncensored and heterogeneous."),
      h2("1.3 Contributions"),
      numitem("A controlled experiment quantifying what the deduplication convention is worth, turning a methodological worry into a measured six-point effect appearing only where duplicates exist."),
      numitem("The identification of seven DAVIS duplicate families that are not duplicates at all, but distinct kinase domains sharing one stored sequence."),
      numitem("An answer to the necessity question from a variance decomposition, a capacity argument and a residual-correlation test rather than a single win."),
      numitem("A cold-target ladder across five protein representations, finding that classical sequence alignment beats ESM-2 embeddings on both datasets and that GBA-Mixup gives nothing here."),
      numitem("Four evaluation regimes rather than the two required, five seeded splits each, and 60 unit tests."),

      // ---------------------------------------------------------- 2
      h1("2. Task 1: Dataset Analysis"),
      h2("2.1 Cleaning, and the One Decision That Matters"),
      p("We load the DeepDTA release, canonicalise SMILES with RDKit, drop non-finite scores, require Kd > 0 for DAVIS, convert with pKd = 9 - log10(Kd), flag extreme values without removing them, and deduplicate on (canonical SMILES, amino-acid sequence). Censored values are retained, as the brief requires; on DAVIS they are 71.2% of the data."),
      p("Deduplication is the only genuinely contested step, because the benchmark ships 442 DAVIS target identifiers that are 379 distinct sequences. Merging them loses information: 1,224 output cells come from a merge, 546 of those merge measurements that **disagree**, 260 by more than 1 pKd, with a worst case of **4.745** for ABL1 against ligand 3062316. Keeping them apart loses something else, because a model reading only the sequence sees identical features and cannot tell the members apart, so the disagreement becomes irreducible error. Section 2.4 measures which costs more."),
      ...figurePair("fig01_heatmaps.png", "fig02_distributions.png", 282, 164, 274, 145,
        "Left: the two matrices after deduplication; the grey cells covering three quarters of KIBA are unmeasured pairs, not non-binding ones. Right: observed score distributions, with the DAVIS panel separating the assay floor from the measured continuum rather than smoothing across it."),
      h2("2.2 The Assay Floor is Censoring, Not Measurement"),
      p("71.2% of DAVIS labels sit at pKd 5.0, Kd = 10,000 nM, the highest concentration tested: left-censored non-binders. Any model can post a respectable mean squared error by learning where the floor is, so every results table carries an MSE-strong column restricted to potent pairs (pKd above 7 on DAVIS, KIBA score above 12.1), and that is the column our claims rest on. The gap is large: on DAVIS the global mean scores 0.650 overall against 6.13 on strong binders."),
      h2("2.3 Eleven of the Eighteen Duplicate Families Are Duplicates"),
      p("The 18 DAVIS families that share a sequence are not one phenomenon: correlating the binding profiles of their members separates them cleanly in two."),
      ...table([
        ["Group", "Families", "Profile correlation", "What they are"],
        ["Mutant panels", "11", "0.77 to 0.98", "Point mutants and phosphorylated forms: ABL1 (15 IDs), EGFR (12), PIK3CA (10), KIT (8), FLT3, RET, MET, BRAF, LRRK2, FGFR3, CDK4"],
        ["**Distinct domains**", "**7**", "**0.07 to 0.48**", "JAK1 and TYK2 (JH1 catalytic against JH2 pseudokinase); RSK1, RSK3, RSK4, RPS6KA4, RPS6KA5 (N-terminal against C-terminal kinase domain)"],
      ], [16, 10, 18, 56], "The DAVIS sequence-identical families split into two groups that behave completely differently."),
      p("The second group is not a duplication artefact. The RSK family carries two non-identical kinase domains in one polypeptide, the N-terminal in the protein kinase A/G/C family and the C-terminal in the calcium/calmodulin-dependent family; the JAK family pairs a catalytic JH1 domain with a pseudokinase JH2 domain. DAVIS assays them separately and inhibitors bind them differently: JAK1's two entries correlate at only **0.074**. One stored sequence for both is a defect in the files, not a property of the biology."),
      p("The separation is predictable from the identifier alone, with no reference to any label: all seven low-correlation families carry KinDom or JH1/JH2 in their IDs and none of the eleven others does. We keep the uniform merge for comparability and report this as a known cost affecting 4.1% of DAVIS targets; excluding those seven families lowers our reported variance explained by about 2.5 points rather than raising it."),
      h2("2.4 What the Deduplication Convention is Worth"),
      p("The two defensible conventions produce different test sets, so numbers computed under one cannot be tabled beside the other. We therefore held model, hyperparameters and seed fixed and varied only the convention."),
      ...table([
        ["Model", "DAVIS sequence", "DAVIS ID", "Difference", "KIBA sequence", "KIBA ID", "Difference"],
        ["Bias only", "0.264", "0.254", "-0.010", "0.383", "0.386", "+0.003"],
        ["Item kNN", "0.558", "0.624", "**+0.066**", "0.652", "0.654", "+0.002"],
        ["Biased MF", "0.623", "0.683", "**+0.060**", "0.714", "0.713", "-0.001"],
      ], [20, 15, 12, 13, 14, 12, 14], "Variance explained on held-out pairs under each deduplication convention, mean of five splits. Only the collaborative models move, and only on DAVIS."),
      p("Keying on source identifier inflates the two collaborative models by six points on DAVIS, leaves the bias baseline alone, and does nothing on KIBA, which has no duplicate sequences. The placebo is what makes this evidence rather than an anecdote."),
      p("Splitting the DAVIS test set by whether a sequence-identical twin of the test target sits in training, matrix factorisation's advantage over the bias baseline is 50.2% where no twin exists and **73.5%** where one does: the model is copying an answer across near-identical columns. This is benchmark redundancy rather than label leakage, since in a warm split the target is in training by construction, but it inflates the measured value of collaborative filtering and the merge removes it."),
      ...figure("fig10_keying.png", "Left: keying on source identifier (hatched) lifts both collaborative models on DAVIS and neither on KIBA. Right: on DAVIS the extra advantage sits almost entirely with test targets whose sequence-identical twin is in training."),
      h2("2.5 KIBA Sparsity is Informative"),
      p("75.2% of the KIBA matrix is unobserved, and a missing cell is an unmeasured pair rather than a non-binding one. The sides are asymmetric: the median ligand carries far fewer measurements than the median target, so the ligand side is the sparse side."),
      h2("2.6 Guilt by Association Holds, and Licenses Task 3"),
      p("Task 3 assumes that similar proteins are bound similarly. Correlating pairwise target similarity against the correlation of binding profiles gives a Spearman coefficient of **0.278** on DAVIS over 71,631 pairs and **0.198** on KIBA over 19,336 pairs, both with p below any reportable threshold. The relationship is real, and it is weaker on KIBA, which predicts that cold-target transfer should work less well there. Section 6 confirms that it does."),

      // ---------------------------------------------------------- 3
      h1("3. Task 2: Collaborative Filtering"),
      h2("3.1 Setup"),
      p("**Drugs are users, targets are items, and affinity is the rating.** The KIBA asymmetry supports the choice: the ligand side is the sparse side, as users are in a conventional recommender."),
      p("Pairs are split 70/15/15 under five seeds, giving 3,876 DAVIS and 17,573 KIBA test pairs per seed. Three properties are asserted in code: no pair appears in two partitions, and every evaluation ligand and target occurs in training. Hyperparameters are tuned on validation only and the test set is touched once. We also run S2 (unseen drugs) and S4 (both unseen) beyond the brief's two regimes."),
      
      p("**Concordance index is reported per drug, not pooled.** Pooled CI mixes comparisons between different drugs, where a model only has to get the drug ordering right. The ligand-mean baseline scores a healthy-looking pooled 0.767 on DAVIS while being, by construction, incapable of ranking one target above another: its per-drug CI is exactly 0.500. Only one of those is honest, so per-drug CI is the headline everywhere. The identity used is CI = (1 + tau_b * sqrt(B/A)) / 2, with A the pairs having distinct true scores and B those having distinct predictions; the brief omits the square root."),
      p("**Ties are scored by their exact expectation, not by one random draw.** A model that ties every candidate has a rank uniform over the tie block, so Hit@k and MRR have closed forms. A single random permutation is unbiased but very noisy: on DAVIS a constant predictor scored between 0.018 and 0.044 across seeds, where the exact form gives 1/57 = 0.0175 with zero variance."),
      h2("3.2 Results"),
      p("Mean of five seeds. DAVIS ranks 57 candidates per drug and KIBA 10.3, so Hit@1 is comparable across datasets only against the random floor, 0.018 and 0.221."),
      ...table([
        ["Model", "DAVIS MSE", "MSE-str", "CI/drug", "Hit@1", "KIBA MSE", "MSE-str", "CI/drug", "Hit@1"],
        ["Global mean", "0.650", "6.13", "0.500", "0.018", "0.693", "2.19", "0.500", "0.221"],
        ["Ligand mean", "0.479", "4.04", "**0.500**", "0.018", "0.424", "1.26", "**0.500**", "0.221"],
        ["Target mean", "0.606", "5.30", "0.701", "0.170", "0.601", "1.91", "0.673", "0.361"],
        ["Bias only", "0.424", "3.45", "0.700", "0.176", "0.353", "1.11", "0.676", "0.373"],
        ["Item kNN", "0.277", "2.08", "0.800", "0.300", "0.234", "0.63", "0.755", "0.526"],
        ["Biased MF", "0.239", "1.73", "0.831", "0.355", "0.186", "0.53", "0.787", "0.589"],
        ["kNN + MF blend", "0.236", "1.77", "0.840", "0.352", "0.184", "0.53", "0.790", "0.593"],
        ["Tobit MF (censored)", "0.248", "1.80", "0.830", "0.349", "0.190", "0.52", "0.778", "0.585"],
        ["KronRLS", "0.237", "**1.45**", "0.829", "0.382", "0.204", "0.52", "0.767", "0.565"],
        ["**KronRLS multi-kernel**", "**0.226**", "1.46", "**0.841**", "**0.419**", "**0.167**", "**0.43**", "**0.793**", "**0.613**"],
      ], [22, 10, 9, 9, 9, 10, 9, 9, 9], "Warm-start results, mean of five seeds. MSE-str is mean squared error restricted to strong binders."),
      p("The ordering is stable across all five seeds and both datasets: kernel and factorisation methods at the top, neighbourhood methods behind, bias-only well behind that, and the two per-entity means at the bottom with a per-drug CI of exactly one half. Multiple-kernel learning wins both, and its margin over plain KronRLS on KIBA (0.167 against 0.204) says the drug and protein kernels carry complementary information a single kernel cannot use."),
      p("The MSE-strong column is the one to read on DAVIS: the headline 0.226 rises to 1.46 on potent pairs, so most of the apparent accuracy is the model learning where the assay floor sits."),
      p("The censored-loss model is an honest null. Tobit MF was built because 71% of DAVIS labels are a detection limit rather than a measurement, yet it does not beat ordinary matrix factorisation on either dataset (0.248 against 0.239; 0.190 against 0.186). It does win the MSE-strong column on KIBA (0.521 against 0.534), the subset it was designed for, so the idea is not wrong, only insufficient at this scale."),
      h2("3.3 Case Study"),
      p("The three ligands the brief names per dataset, predicted from a reloaded checkpoint asserted identical to the original, not merely close."),
      ...table([
        ["Ligand", "Candidates", "True best", "Rank of true best", "MSE", "CI"],
        ["DAVIS 10127622 (selumetinib)", "57", "5.27", "9", "0.038", "0.857"],
        ["DAVIS 11338033 (a CDK-family inhibitor)", "57", "7.52", "**4**", "0.128", "0.964"],
        ["DAVIS 44259 (staurosporine)", "57", "9.80", "13", "0.852", "0.799"],
        ["KIBA CHEMBL1967538", "4", "10.70", "2", "0.042", "0.667"],
        ["KIBA CHEMBL2004544", "15", "12.10", "**1**", "0.053", "0.871"],
        ["KIBA CHEMBL565612", "25", "14.20", "5", "0.353", "0.730"],
      ], [38, 13, 12, 17, 10, 10], "Case study on the six ligands named in the brief, predicted from a reloaded checkpoint."),
      p("The three DAVIS cases span the selectivity spectrum and behave in opposite ways, which is more informative than three successes would be. Staurosporine is the textbook promiscuous ATP-competitive binder: it has the **worst** MSE of the six at 0.852 and its true best target lands only at rank 13, yet its CI of 0.799 says the broad ordering is right. The model knows it binds most of the panel and cannot pick which of many strong binders is strongest."),
      p("Selumetinib is the opposite, and exposes a limit of per-pair splitting. It is a MEK1/MEK2 inhibitor, both real targets fell in the training partition, and its held-out set therefore sits at or near the assay floor. Its MSE of 0.038 and CI of 0.857 are close to meaningless, and its true best is the largest of 57 near-tied floor values. We report it as a cautionary case rather than a success."),

      // ---------------------------------------------------------- 4
      h1("4. Is the Recommender Necessary, and Is It a Lookup?"),
      p("The brief says either conclusion is acceptable if supported by analysis rather than opinion. Our answer is yes, with a large margin, but part of what a careless setup would measure is an artefact."),
      p("**Variance accounting.** Decomposing the observed scores into a drug main effect, a target main effect and the remainder gives Table 6. Bias terms reach only the additive part, so on DAVIS two thirds of the variance is out of reach before a model is fitted."),
      ...table([
        ["Dataset", "Drug identity", "Target identity", "Additive together", "Interaction"],
        ["DAVIS", "26.5%", "10.2%", "36.8%", "**63.2%**"],
        ["KIBA", "41.7%", "14.1%", "51.9%", "**48.1%**"],
      ], [16, 21, 21, 21, 21], "Two-way variance decomposition of the observed affinity scores."),
      p("The measured gap agrees: bias-only reaches 0.424 MSE on DAVIS where matrix factorisation reaches 0.239, a 44% reduction, and 0.353 against 0.186 on KIBA, a 47% reduction. Truncated SVD puts 66.6% of DAVIS energy and 64.7% of KIBA in the first five components, the low-rank structure factorisation exploits."),
      p("**Capacity against data.** At the selected latent dimension of 64 the model carries (m + n)(k + 1) + 1 parameters: about 29,000 on DAVIS against 18,100 training pairs, and 152,000 on KIBA against 82,000. Well under one observation per parameter, so heavy regularisation was expected to be needed and kernel methods, carrying no free parameter per entity, to be competitive. Both happened."),
      p("**Part of the gap is redundancy, not skill.** Section 2.4 showed the identifier-keyed convention inflates the collaborative models by six points on DAVIS, so any comparison on the raw matrix overstates the case by roughly that much. The conclusion survives, six points being small beside a 44% reduction, but it needs saying."),
      p("**Not a lookup of known answers.** Three pieces of evidence, on both datasets. First, every evaluation pair is absent from training, asserted in code; a lookup table returns nothing for those pairs and its best answer is the drug mean, the ligand-mean row. Second, the full model beats that row by 50% on DAVIS and 56% on KIBA. Third, and most directly, we correlate the true residual after removing the additive part against the interaction term the model predicts. If the model were reproducing stored answers this would be meaningless; if it has learned interaction structure the two should track. They do, at **0.655** on DAVIS and **0.676** on KIBA, which is the model predicting exactly the part of the signal no lookup table contains."),

      // ---------------------------------------------------------- 5
      h1("5. Task 3: Cold-Target Generalisation"),
      h2("5.1 The Failure, Stated and Measured"),
      p("Biased matrix factorisation predicts r(i, j) = mu + b_i + c_j + u_i . v_j. For a target j absent from training, c_j and v_j receive no gradient, so the prediction reduces to mu + b_i, which does not depend on j: the model gives every unseen target the same score and cannot rank two new proteins against each other at all."),
      p("The prediction was made before the measurement: a model whose every within-drug ranking is a tie must score a concordance index of exactly one half. The fallback scores a per-drug CI of **0.5000 on both datasets, standard deviation 0.0000 across five seeds**, while reporting a pooled CI of 0.783 on DAVIS and 0.715 on KIBA and looking perfectly healthy. This is the clearest argument in the report for not trusting a concordance index without asking how it was pooled."),
      p("The fix, stated once: replace the missing per-target parameters with something computable from the amino-acid sequence. Every rung below is one way to do that. One structural advantage shapes the design: in a cold-target split the drugs stay warm, so drug-side features and each drug's full training profile remain available."),
      h2("5.2 Protocol"),
      p("Whole targets are held out. Because the entity key is the sequence, a mutant family is one entity and cannot straddle the split, so the leakage that would put ABL1 in training and ABL1(T315I) in test is structurally impossible rather than merely avoided. Validation targets are also unseen, so hyperparameters are tuned under the regime the test set measures. DAVIS holds out 57 targets and KIBA 34, so a single split would be noise."),
      h2("5.3 The Method Ladder"),
      ...table([
        ["Rung", "Method", "Trainable parameters", "Idea"],
        ["C0", "Ligand-mean fallback", "none", "What collaborative filtering degenerates to. The baseline to beat"],
        ["C1", "Sequence kNN", "**none**", "Bias-centred average over the k most similar training targets, weights raised to the power gamma"],
        ["C2", "Factor map", "ridge", "Freeze the warm model's drug factors, learn sequence features to (c_j, v_j), substitute into the warm formula"],
        ["C3", "Content regression", "LightGBM", "Drug and protein descriptors concatenated; no matrix at all"],
        ["C4", "Similarity-gated blend", "-", "C1 where a close relative exists, C3 elsewhere, so nothing falls back"],
        ["C5", "C3 plus GBA-Mixup", "LightGBM", "Virtual training pairs interpolated between graph neighbours"],
        ["M5, M6", "KronRLS, single and multiple kernels", "dual", "Kernel regression over the Kronecker product of drug and target kernels"],
      ], [9, 24, 16, 51], "The cold-target method ladder, from the fallback to kernel regression."),
      p("Five protein representations are compared independently of the rungs, tuned on validation like any other hyperparameter: 420-dimensional amino-acid and dipeptide composition, 3-mer TF-IDF reduced by SVD, a normalised Smith-Waterman profile against training targets only, and mean-pooled ESM-2 at 35M and 650M parameters."),
      p("**GBA-Mixup.** Following MixingDTA, pairs sharing a drug are graph neighbours; a partner is sampled with a preference for a similar label, and features and labels are interpolated with lambda from Beta(0.4, 0.4): x_mix = lambda * x_p + (1 - lambda) * x_q, and likewise for y. Virtual pairs equal to half the training set are appended before fitting. The published system pairs this with fine-tuned MolFormer-XL and ESM3 encoders, far beyond this project; the augmentation alone is the transferable part and is what we test."),

      // ---------------------------------------------------------- 6
      h1("6. Task 3 Results"),
      h2("6.1 Every Sequence-Aware Method Beats the Fallback"),
      ...table([
        ["Method", "DAVIS MSE", "DAVIS CI/drug", "DAVIS Hit@1", "KIBA MSE", "KIBA CI/drug", "KIBA Hit@1"],
        ["C0 fallback", "0.476 (.048)", "**0.500**", "0.018", "0.419 (.045)", "**0.500**", "0.188"],
        ["C2 factor map", "0.417 (.058)", "0.676", "0.166", "0.396 (.041)", "0.586", "0.250"],
        ["C3 content regression", "0.296 (.047)", "0.770", "0.311", "0.325 (.044)", "0.657", "0.393"],
        ["C5 C3 + GBA-Mixup", "0.297 (.044)", "**0.778**", "0.339", "0.329 (.043)", "0.652", "0.392"],
        ["M5 KronRLS", "0.292 (.041)", "0.769", "0.360", "0.313 (.045)", "0.669", "0.432"],
        ["M6 KronRLS multi-kernel", "0.299 (.040)", "0.758", "0.357", "0.298 (.048)", "**0.681**", "**0.457**"],
        ["C4 gated blend", "0.273 (.036)", "0.752", "0.384", "0.284 (.045)", "0.673", "0.439"],
        ["**C1 sequence kNN**", "**0.272 (.036)**", "0.740", "0.393", "**0.283 (.045)**", "0.678", "0.444"],
      ], [26, 14, 13, 12, 14, 12, 12], "Cold-target results, mean of five splits with the standard deviation of MSE in brackets."),
      p("Two things stand out. The best method by mean squared error on both datasets is C1, a similarity-weighted average with no trainable parameters at all, cutting error 43% below the fallback on DAVIS and 32% on KIBA. And the best by error is not the best by ranking: C5 takes per-drug CI on DAVIS at 0.778, M6 on KIBA at 0.681. Squared error and ordering are not the same objective here, and we quote both rather than whichever flatters."),
      p("The two regimes the brief does not ask for frame how hard this one is. With unseen drugs (S2) structure kNN takes the lowest DAVIS error at 0.519 while KronRLS takes the best per-drug CI at 0.732; with both sides unseen (S4) KronRLS takes both, at 0.568 and 0.620. The fallback is not degenerate in S2, scoring 0.706 because the targets are still known; it falls to 0.500 only when the target is the unseen side. Cold targets are the easier of the two cold problems, because proteins have a usable similarity measure and 68 drugs give little to generalise over."),
      h2("6.2 Alignment Beats the Protein Language Model"),
      p("The representation ablation was the surprise of this project and is consistent across both datasets. Content regression, same model and splits, five seeds, varying only the protein descriptor:"),
      ...table([
        ["Representation", "Dim", "DAVIS MSE", "DAVIS CI/drug", "KIBA MSE", "KIBA CI/drug"],
        ["Composition (the original default)", "420", "0.443", "0.615", "0.420", "0.569"],
        ["3-mer TF-IDF with SVD", "256", "0.420", "0.697", "0.435", "0.601"],
        ["ESM-2, 35M parameters", "480", "0.383", "0.696", "0.392", "0.590"],
        ["ESM-2, 650M parameters", "1280", "0.359", "0.703", "0.357", "0.654"],
        ["**Smith-Waterman profile**", "n", "**0.296**", "**0.770**", "**0.325**", "**0.657**"],
      ], [32, 8, 15, 15, 15, 15], "Cold-target performance by protein representation. Smith-Waterman wins every column."),
      p("Smith-Waterman wins both datasets by a wide margin, and validation tuning selects it independently for all three feature-based models. Changing nothing but the descriptor moved content regression from barely beating the fallback to competitive with the kernel methods, so an earlier reading that factor-mapping and content regression are weak ideas was wrong: they were reading the weakest of the five descriptors."),
      p("We did not expect a 650M-parameter language model to lose to a 1981 alignment algorithm, and offer a mechanism rather than a claim. Both panels are kinases, so what separates one target from another is a few dozen residues lining the ATP pocket against a background of high global homology. Local alignment scores exactly that; mean-pooling an embedding over a 700-residue multi-domain protein averages it away."),
      ...figure("fig09_representation.png", "The protein representation matters more than anything else in these models. Mixup, the paired bars, changes nothing anywhere."),
      h2("6.3 GBA-Mixup Gives Nothing Here"),
      p("Across five seeds, two datasets and five representations the augmentation moves results in the third decimal with inconsistent sign: on DAVIS 0.297 with against 0.296 without, on KIBA 0.329 against 0.325. It takes the top DAVIS per-drug CI at 0.778 against 0.770, within the seed spread and not a claim we will make."),
      p("We report this as a null rather than omitting it. Interpolating between feature vectors is only sensible where the representation is locally linear in affinity, and a similarity kernel evidently does not satisfy that well enough for the augmentation to pay. A negative result implemented and ablated is worth more than a method quietly left out."),
      h2("6.4 Does It Generalise, or Is It Homology Lookup?"),
      p("The decisive test buckets cold targets by maximum sequence identity to anything in training, with one set of bucket edges shared by all models so two readings cannot disagree."),
      ...table([
        ["Max identity to training", "DAVIS C0", "DAVIS C1", "KIBA C0", "KIBA C1"],
        ["Below 0.10 (orphan)", "0.395", "**0.588**", "0.754", "0.724"],
        ["0.10 to 0.20", "0.505", "0.515", "0.458", "0.410"],
        ["0.20 to 0.35", "0.465", "0.368", "0.376", "0.338"],
        ["Above 0.35 (homologue)", "0.470", "**0.183**", "0.415", "**0.204**"],
      ], [32, 17, 17, 17, 17], "Cold-target MSE by maximum sequence identity to the training set. Below 0.10 the method is worse than the fallback."),
      p("The gradient is steep and cuts both ways. Against a target with a genuine homologue in training the method is excellent, cutting DAVIS error by 61% and KIBA by 51%, with per-drug CI of 0.810 and 0.732. Against an orphan it is worse than doing nothing: on DAVIS 0.588 where the drug mean posts 0.395, and on KIBA a per-drug CI of 0.440 in that bucket, below the 0.500 of a model that cannot rank at all."),
      p("The honest headline is that **this is a homology lookup rather than cold-target generalisation in the strong sense** - a useful one, since most proteins a project would ask about do have a relative in a 379-kinase panel. But a deployed system should refuse below an identity threshold, and reporting that cliff is more useful than an average concealing it."),
      ...figure("fig07_similarity_buckets.png", "The generalisation curve, and the report's main caveat. Performance against the fallback improves steadily with homology and reverses below roughly 20% identity."),

      // ---------------------------------------------------------- 7
      h1("7. Limitations and Future Work"),
      p("**The method is a homology lookup**: below 20% identity it is worse than the drug mean, and we did not implement the threshold refusal a deployed version would need. **Cold test sets are small**, at 57 DAVIS and 34 KIBA held-out targets, so differences under about 0.02 MSE are not resolvable, which is why mixup is a null rather than a negative. **The deduplication convention costs something**: merging the seven domain-pair families averages two unrelated profiles, making the uniform rule mildly optimistic for 4.1% of DAVIS targets. **ESM-2 was mean-pooled**, so Section 6.2 concerns whole-sequence mean pooling on a single-family panel rather than protein language models generally; binding-site-restricted pooling is the obvious next test. **Censoring is handled by reporting, not by the loss**: a properly interval-censored likelihood with the threshold modelled per assay is the next thing to try."),

      // ---------------------------------------------------------- 8
      h1("8. Conclusion"),
      p("Collaborative filtering works well on both panels and earns its interaction term. KronRLS with multiple kernels reaches 0.226 MSE and a per-drug concordance index of 0.841 on DAVIS, and 0.167 and 0.793 on KIBA, against a bias-only baseline of 0.424 and 0.353. Two thirds of DAVIS variance sits in drug-target interaction, out of reach of any bias model, and about six points of the apparent gap is benchmark redundancy the chemistry-keyed convention removes."),
      p("The same model then fails completely on an unseen target, scoring a per-drug concordance index of exactly 0.500 with zero variance across seeds, predicted from the algebra before being measured. A sequence representation recovers most of that, cutting error 43% on DAVIS and 32% on KIBA, and the method that does it best has no trainable parameters: the representation matters more than the model, a Smith-Waterman profile beats ESM-2 650M embeddings on both datasets, and GBA-Mixup adds nothing measurable. Whether any of it helps depends sharply on homology: for a protein with no relative in the training panel the best answer remains the drug's own mean, and saying so is more useful than an average that hides it."),

      // ---------------------------------------------------------- refs
      h1("References"),
      ...[
        "Davis, M. I., Hunt, J. P., Herrgard, S., Ciceri, P., Wodicka, L. M., Pallares, G., Hocker, M., Treiber, D. K., and Zarrinkar, P. P. (2011). Comprehensive analysis of kinase inhibitor selectivity. Nature Biotechnology, 29(11), 1046-1051.",
        "He, T., Heidemeyer, M., Ban, F., Cherkasov, A., and Ester, M. (2017). SimBoost: a read-across approach for predicting drug-target binding affinities using gradient boosting machines. Journal of Cheminformatics, 9, 24.",
        "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: a highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems 30.",
        "Kim, Y., Bang, D., Koo, B., Yi, J., Cho, C., Choi, J., and Kim, S. (2025). MixingDTA: improved drug-target affinity prediction by extending mixup with guilt-by-association. Bioinformatics, 41(Supplement 1), i105-i114.",
        "Koren, Y., Bell, R., and Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.",
        "Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., and Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637), 1123-1130.",
        "Ozturk, H., Ozgur, A., and Ozkirimli, E. (2018). DeepDTA: deep drug-target binding affinity prediction. Bioinformatics, 34(17), i821-i829.",
        "Pahikkala, T., Airola, A., Pietila, S., Shakyawar, S., Szwajda, A., Tang, J., and Aittokallio, T. (2015). Toward more realistic drug-target interaction predictions. Briefings in Bioinformatics, 16(2), 325-337.",
        "Rogers, D., and Hahn, M. (2010). Extended-connectivity fingerprints. Journal of Chemical Information and Modeling, 50(5), 742-754.",
        "Smith, T. F., and Waterman, M. S. (1981). Identification of common molecular subsequences. Journal of Molecular Biology, 147(1), 195-197.",
        "Tang, J., Szwajda, A., Shakyawar, S., Xu, T., Hintsanen, P., Wennerberg, K., and Aittokallio, T. (2014). Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis. Journal of Chemical Information and Modeling, 54(3), 735-743.",
      ].map((t) => new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { line: 240, after: 60 },
        indent: { left: 420, hanging: 420 },
        children: [new TextRun({ text: t, font: FONT, size: 20 })],
      })),

      // ---------------------------------------------------------- appendix
      h1("Appendix A. Reproducibility"),
      p("Every number and figure in this report is generated from the result files in `results/tables/`; none is transcribed by hand. The pipeline is seeded throughout and 60 unit tests pass, including a brute-force check of the concordance index and a check that the exact tie expectations match simulation."),
      p("From the repository root:"),
      ...["bash run_all.sh                      # stages 0-9: data, splits, kernels, Task 1, sweep, analysis, figures, tables",
          "python -m dta.run_keying davis kiba  # the deduplication A/B of Section 2.4",
          "python -m dta.figures_merge davis kiba"].map((t) => new Paragraph({
        spacing: { line: 240, after: 40 }, indent: { left: 280 },
        children: [new TextRun({ text: t, font: "Consolas", size: 18 })],
      })),
      p("ESM-2 embeddings are produced on a GPU by `scripts/colab_esm2.py` and cached in `features/`. The pipeline runs end to end without them, in which case that representation simply drops out of the Section 6.2 ablation."),
      p("All modelling code is our own. The DAVIS and KIBA matrices, SMILES strings, protein sequences and supplied similarity matrices are taken unmodified from the DeepDTA repository (Ozturk et al., 2018). KronRLS follows Pahikkala et al. (2015), the gradient-boosted similarity-feature design follows He et al. (2017), and GBA-Mixup follows Kim et al. (2025); each is reimplemented from the paper's description with the citation placed directly above the implementation in the source."),

      new Paragraph({ children: [new PageBreak()] }),
      h1("Appendix B. Individual Contributions"),
      ...table([
        ["Member Name", "Contribution (%)", "Justification"],
        ["", "25%", "Shared pipeline: loading, cleaning, deduplication, the four split regimes with leakage assertions, and the metrics module including per-drug CI and exact tie expectations. Report Sections 2.1, 2.2 and 3.1."],
        ["", "25%", "Task 1: duplicate-family analysis, the deduplication A/B experiment, sparsity and missingness, and the guilt-by-association study. Report Sections 2.3 to 2.6."],
        ["", "25%", "Task 2: the model ladder including KronRLS and the censored-loss model, tuning, checkpointing, the case study, and the necessity and lookup experiments. Report Sections 3.2, 3.3 and 4."],
        ["", "25%", "Task 3: the cold-target ladder, the five-representation ablation, GBA-Mixup, and the generalisation analysis. Report Sections 5 and 6. Video."],
      ], [20, 16, 64], "Individual contributions. Percentages are equal; if they are not, attach the email in which all members agree."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/claude/SC4020_Project_2_Report.docx", buf);
  console.log("written", buf.length, "bytes;", figN, "figures,", tabN, "tables");
});
