import { useEffect, useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  ChevronRight,
  Database,
  Gauge,
  Home,
  Image as ImageIcon,
  Layers3,
  Menu,
  ShieldCheck,
  Target,
  X,
} from "lucide-react";

const pages = {
  dashboard: "Dashboard",
  inspection: "Inspection",
  analytics: "Analytics",
  results: "Results",
  heatmaps: "Heatmaps",
};

const charts = [
  {
    title: "Image-Level AUROC",
    file: "/results/01_auroc.png",
    description: "Measured anomaly detection performance across 15 MVTec AD categories.",
  },
  {
    title: "Average Precision",
    file: "/results/02_average_precision.png",
    description: "Measured precision-recall performance across the evaluated categories.",
  },
  {
    title: "Detection Comparison",
    file: "/results/03_detection_comparison.png",
    description: "AUROC and average precision comparison for each category.",
  },
  {
    title: "Defect Localization",
    file: "/results/04_localization.png",
    description: "Pixel-level localization performance measured using pixel AUROC.",
  },
  {
    title: "Overall Performance",
    file: "/results/05_overall_performance.png",
    description: "Aggregate measured detection performance.",
  },
];

const heatmaps = [
  ["Bottle", "bottle_broken_large_000.png", "Broken Large"],
  ["Cable", "cable_bent_wire_000.png", "Bent Wire"],
  ["Capsule", "capsule_crack_000.png", "Crack"],
  ["Carpet", "carpet_color_000.png", "Color"],
  ["Grid", "grid_bent_000.png", "Bent"],
  ["Hazelnut", "hazelnut_crack_000.png", "Crack"],
  ["Leather", "leather_color_000.png", "Color"],
  ["Metal Nut", "metal_nut_bent_000.png", "Bent"],
  ["Pill", "pill_color_000.png", "Color"],
  ["Screw", "screw_manipulated_front_000.png", "Manipulated Front"],
  ["Tile", "tile_crack_000.png", "Crack"],
  ["Toothbrush", "toothbrush_defective_000.png", "Defective"],
  ["Transistor", "transistor_bent_lead_000.png", "Bent Lead"],
  ["Wood", "wood_color_000.png", "Color"],
  ["Zipper", "zipper_broken_teeth_000.png", "Broken Teeth"],
];

function App() {
  const [page, setPage] = useState("dashboard");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [selectedHeatmap, setSelectedHeatmap] = useState(heatmaps[0]);

  const navigate = (nextPage) => {
    setPage(nextPage);
    setMobileOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="app-shell">
      <Sidebar
        page={page}
        navigate={navigate}
        mobileOpen={mobileOpen}
      />

      <main className="main-area">
        <Topbar
          page={page}
          mobileOpen={mobileOpen}
          setMobileOpen={setMobileOpen}
        />

        {page === "dashboard" && <Dashboard navigate={navigate} />}

        {page === "inspection" && <Inspection />}

        {page === "analytics" && <Analytics />}

        {page === "results" && <Results navigate={navigate} />}

        {page === "heatmaps" && (
          <HeatmapExplorer
            selected={selectedHeatmap}
            setSelected={setSelectedHeatmap}
          />
        )}
      </main>
    </div>
  );
}

function Sidebar({ page, navigate, mobileOpen }) {
  return (
    <aside className={`sidebar ${mobileOpen ? "mobile-open" : ""}`}>
      <div className="brand">
        <div className="brand-icon">
          <ShieldCheck size={23} strokeWidth={2} />
        </div>

        <div>
          <div className="brand-title">
            SENTINEL<span>VISION</span>
          </div>
          <div className="brand-caption">VISUAL QUALITY INTELLIGENCE</div>
        </div>
      </div>

      <div className="nav-heading">WORKSPACE</div>

      <nav className="navigation">
        <NavButton
          icon={<Home size={18} />}
          label="Dashboard"
          active={page === "dashboard"}
          onClick={() => navigate("dashboard")}
        />

        <NavButton
          icon={<Activity size={18} />}
          label="Inspection"
          active={page === "inspection"}
          onClick={() => navigate("inspection")}
        />

        <NavButton
          icon={<BarChart3 size={18} />}
          label="Analytics"
          active={page === "analytics"}
          onClick={() => navigate("analytics")}
        />

        <NavButton
          icon={<Activity size={18} />}
          label="Evaluation Results"
          active={page === "results"}
          onClick={() => navigate("results")}
        />

        <NavButton
          icon={<ImageIcon size={18} />}
          label="Defect Heatmaps"
          active={page === "heatmaps"}
          onClick={() => navigate("heatmaps")}
        />
      </nav>

      <div className="sidebar-footer">
        <button type="button" className="status-card" onClick={() => navigate("results")}>
          <div className="status-indicator" />
          <div>
            <strong>Evaluation Ready</strong>
            <span>Local project environment</span>
          </div>
        </button>

        <button type="button" className="model-card" onClick={() => navigate("inspection")}>
          <div className="model-icon">
            <BrainCircuit size={18} />
          </div>
          <div>
            <strong>PatchCore</strong>
            <span>WideResNet-50-2 + FAISS</span>
          </div>
        </button>
      </div>
    </aside>
  );
}

function NavButton({ icon, label, active, onClick }) {
  return (
    <button
      type="button"
      className={`nav-button ${active ? "active" : ""}`}
      onClick={onClick}
    >
      {icon}
      <span>{label}</span>
      {active && <span className="active-marker" />}
    </button>
  );
}

function Topbar({ page, mobileOpen, setMobileOpen }) {
  return (
    <header className="topbar">
      <button
        type="button"
        className="mobile-menu"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Open navigation"
      >
        {mobileOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      <div className="breadcrumbs">
        <span>SENTINELVISION</span>
        <ChevronRight size={14} />
        <strong>{pages[page].toUpperCase()}</strong>
      </div>

    </header>
  );
}

function Dashboard({ navigate }) {
  return (
    <div className="page">
      <PageIntro
        eyebrow="SENTINELVISION"
        title="Visual Anomaly Intelligence"
        description="AI-powered inspection for detecting and localizing manufacturing defects."
      />

      <div className="metric-grid">
        <MetricCard
          icon={<Target size={20} />}
          label="Mean Image AUROC"
          value="94.39%"
          detail="Across evaluated categories"
        />
        <MetricCard
          icon={<Gauge size={20} />}
          label="Mean Average Precision"
          value="97.58%"
          detail="Across evaluated categories"
        />
        <MetricCard
          icon={<ImageIcon size={20} />}
          label="Mean Pixel AUROC"
          value="96.00%"
          detail="Defect localization"
        />
        <MetricCard
          icon={<Database size={20} />}
          label="Product Categories"
          value="15"
          detail="MVTec AD categories"
        />
      </div>

      <div className="dashboard-grid">
        <div className="glass-card">
          <div className="card-title">
            <BrainCircuit size={18} />
            <span>Inspection Pipeline</span>
          </div>

          <div className="pipeline">
            <div className="pipeline-step">
              <span>01</span>
              <strong>Learn Normal Patterns</strong>
              <small>WideResNet-50-2 extracts deep visual features from defect-free products.</small>
            </div>
            <ChevronRight size={20} />
            <div className="pipeline-step">
              <span>02</span>
              <strong>Detect Anomalies</strong>
              <small>PatchCore compares image patches against the learned normal feature memory using FAISS.</small>
            </div>
            <ChevronRight size={20} />
            <div className="pipeline-step">
              <span>03</span>
              <strong>Localize Defects</strong>
              <small>Anomaly scores are mapped back to the image to generate a defect heatmap.</small>
            </div>
          </div>
        </div>

        <div className="glass-card">
          <div className="card-title">
            <ShieldCheck size={18} />
            <span>AI Inspection</span>
          </div>

          <p className="card-description">
            Upload a product image and SentinelVision automatically identifies
            the product category before running anomaly detection.
          </p>

          <button
            className="primary-button"
            onClick={() => navigate("inspection")}
          >
            Start Inspection
            <ChevronRight size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}

function Inspection() {
  const [file, setFile] = useState(null);
  const [category, setCategory] = useState("bottle");
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [cameraOpen, setCameraOpen] = useState(false);
  const videoRef = useRef(null);
  const cameraStreamRef = useRef(null);

  const stopCamera = () => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOpen(false);
  };

  const startCamera = async () => {
    try {
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      cameraStreamRef.current = stream;
      setCameraOpen(true);
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
      }, 50);
    } catch (err) {
      setError("Camera access was denied or is unavailable. Please allow camera access and try again.");
    }
  };

  const captureFromCamera = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const capturedFile = new File([blob], `webcam-${Date.now()}.jpg`, { type: "image/jpeg" });
      handleFile(capturedFile);
      stopCamera();
    }, "image/jpeg", 0.92);
  };

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const runInspection = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("category", category);

      const response = await fetch("http://127.0.0.1:8000/api/inspect", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "object"
            ? data.detail.message
            : (data.detail || "Inspection failed")
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        typeof err.message === "string"
          ? err.message
          : "Unable to connect to SentinelVision API"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="eyebrow">AI QUALITY INSPECTION</div>
          <h1>Inspection</h1>
          <p>Upload a product image or capture one with the webcam, then inspect it using the selected product category.</p>
        </div>
      </div>

      <div className="inspection-grid">
        <div className="glass-card">
          <div className="card-title">
            <ImageIcon size={18} />
            <span>Product Image</span>
          </div>

          <label className="upload-zone">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => handleFile(e.target.files?.[0])}
              hidden
            />

            {preview ? (
              <img
                src={preview}
                alt="Selected product"
                className="inspection-preview"
              />
            ) : (
              <div className="upload-content">
                <ImageIcon size={42} />
                <strong>Upload product image</strong>
                <span>PNG, JPG or JPEG</span>
              </div>
            )}
          </label>

          {file && (
            <div className="file-name">
              {file.name}
            </div>
          )}

          <div className="camera-section">
            <div className="camera-header">
              <div className="card-title">
                <Activity size={18} />
                <span>Webcam Inspection</span>
              </div>
              <span className="camera-badge">LIVE CAPTURE</span>
            </div>

            {cameraOpen ? (
              <div className="camera-preview-wrap">
                <video ref={videoRef} className="camera-preview" autoPlay playsInline muted />
                <div className="camera-controls">
                  <button className="primary-button camera-capture" onClick={captureFromCamera}>
                    Capture Image
                    <ChevronRight size={18} />
                  </button>
                  <button className="secondary-button" onClick={stopCamera}>
                    <X size={17} />
                    Close Camera
                  </button>
                </div>
              </div>
            ) : (
              <button className="camera-open-button" onClick={startCamera}>
                <Activity size={20} />
                <span>
                  <strong>Open Camera</strong>
                  <small>Capture a product directly from your webcam</small>
                </span>
              </button>
            )}
          </div>

          <div className="category-selector">
            <div className="card-title">
              <Target size={18} />
              <span>Product Category</span>
            </div>

            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="category-select"
            >
              <option value="bottle">Bottle</option>
              <option value="cable">Cable</option>
              <option value="capsule">Capsule</option>
              <option value="carpet">Carpet</option>
              <option value="grid">Grid</option>
              <option value="hazelnut">Hazelnut</option>
              <option value="leather">Leather</option>
              <option value="metal_nut">Metal Nut</option>
              <option value="pill">Pill</option>
              <option value="screw">Screw</option>
              <option value="tile">Tile</option>
              <option value="toothbrush">Toothbrush</option>
              <option value="transistor">Transistor</option>
              <option value="wood">Wood</option>
              <option value="zipper">Zipper</option>
            </select>

            <small>
              PatchCore will evaluate the image using only this product category.
            </small>
          </div>

          <button
            className="primary-button"
            onClick={runInspection}
            disabled={!file || loading}
          >
            {loading ? "Analyzing..." : "Run AI Inspection"}
            {!loading && <ChevronRight size={18} />}
          </button>

          {error && (
            <div className="error-box">
              {error}
            </div>
          )}
        </div>

        <div className="glass-card result-panel">
          <div className="card-title">
            <BrainCircuit size={18} />
            <span>AI Inspection Result</span>
          </div>

          {!result && !loading && (
            <div className="empty-result">
              <Target size={42} />
              <strong>Ready for inspection</strong>
              <span>
                SentinelVision will calculate the anomaly score and localize defects using the selected product category.
              </span>
            </div>
          )}

          {loading && (
            <div className="empty-result">
              <Activity size={42} />
              <strong>Analyzing image...</strong>
              <span>
                Running category-specific PatchCore inference.
              </span>
            </div>
          )}

          {result && (
            <div className="inspection-result">
              <div className="detected-product">
                <span>SELECTED PRODUCT</span>
                <strong>{result.predicted_category || result.category}</strong>
                <small>
                  Category-specific PatchCore evaluation
                </small>
              </div>

              <div className={`status-card ${result.status === "ANOMALY" ? "anomaly" : "normal"}`}>
                <span>INSPECTION STATUS</span>
                <strong>{result.status}</strong>
              </div>

              <div className="metric-grid">
                <div className="metric-card">
                  <span>Anomaly Score</span>
                  <strong>{Number(result.score).toFixed(2)}</strong>
                </div>

                <div className="metric-card">
                  <span>Threshold</span>
                  <strong>{Number(result.threshold).toFixed(2)}</strong>
                </div>

                <div className="metric-card">
                  <span>Above Threshold</span>
                  <strong>{Number(result.deviation).toFixed(2)}%</strong>
                </div>

                <div className="metric-card">
                  <span>Patch Count</span>
                  <strong>{result.patch_count}</strong>
                </div>
              </div>

              <ImageAnalysisCharts result={result} />

              <div className="manufacturing-plan">
                <div className="plan-header">
                  <div>
                    <span className="plan-eyebrow">QUALITY CONTROL</span>
                    <h3>AI Manufacturing Recommendations</h3>
                  </div>
                  <span className={`plan-status ${result.status === "ANOMALY" ? "action-required" : "review-complete"}`}>
                    {result.status === "ANOMALY" ? "ACTION REQUIRED" : "REVIEW COMPLETE"}
                  </span>
                </div>

                {(() => {
                  const category = (result.category || "").toLowerCase();
                  const score = Number(result.score || 0);
                  const threshold = Number(result.threshold || 1);
                  const ratio = score / threshold;
                  const excess = Number(result.deviation || 0);
                  const patchRatio = Number(result.high_score_patch_ratio || 0);
                  const patchCount = Number(result.high_score_patch_count || 0);
                  const concentration = result.anomaly_concentration || "LIMITED";

                  let severity = "LOW";

                  if (
                    ratio >= 3 ||
                    patchRatio >= 10 ||
                    concentration === "WIDESPREAD"
                  ) {
                    severity = "CRITICAL";
                  } else if (
                    ratio >= 2 ||
                    patchRatio >= 5 ||
                    concentration === "SIGNIFICANT"
                  ) {
                    severity = "HIGH";
                  } else if (
                    ratio >= 1.25 ||
                    patchRatio >= 2 ||
                    concentration === "LOCALIZED"
                  ) {
                    severity = "MODERATE";
                  }

                  const categoryAdvice = {
                    bottle: {
                      defect: severity === "CRITICAL"
                        ? "Inspect the bottle body, neck and base for cracks, deformation, leakage or structural damage."
                        : "Inspect the bottle surface, neck and closure for visible cracks, deformation or leakage.",
                      verify: "Perform a visual and leak check before the unit is released."
                    },
                    cable: {
                      defect: severity === "CRITICAL"
                        ? "Inspect insulation, exposed conductors, severe bends and connector regions for physical damage."
                        : "Inspect cable insulation, bends and connector regions for abnormal damage.",
                      verify: "Check insulation continuity and connector condition before release."
                    },
                    capsule: {
                      defect: "Inspect the capsule shell for cracks, deformation, surface damage or abnormal shape.",
                      verify: "Confirm the capsule shell shape and surface condition before release."
                    },
                    carpet: {
                      defect: "Inspect the carpet surface for stains, texture changes, holes, cuts or abnormal patterns.",
                      verify: "Check surface texture and appearance under consistent lighting."
                    },
                    grid: {
                      defect: "Inspect grid elements for bending, breakage, missing sections or irregular spacing.",
                      verify: "Check grid alignment, spacing and structural integrity."
                    },
                    hazelnut: {
                      defect: "Inspect the shell for cracks, holes, cuts or abnormal surface damage.",
                      verify: "Perform a detailed visual surface inspection."
                    },
                    leather: {
                      defect: "Inspect the leather surface for scratches, stains, color variation, cuts or texture defects.",
                      verify: "Check texture and surface consistency before release."
                    },
                    metal_nut: {
                      defect: "Inspect the nut for deformation, damaged threads, surface defects or abnormal geometry.",
                      verify: "Check thread condition and dimensional geometry."
                    },
                    pill: {
                      defect: "Inspect the pill for cracks, chips, discoloration, contamination or abnormal surface appearance.",
                      verify: "Check shape, color and surface condition."
                    },
                    screw: {
                      defect: "Inspect the screw head, shaft and threads for deformation, damage or abnormal geometry.",
                      verify: "Check head integrity and thread condition."
                    },
                    tile: {
                      defect: "Inspect the tile for cracks, chips, scratches, surface contamination or pattern abnormalities.",
                      verify: "Check the complete surface and edges."
                    },
                    toothbrush: {
                      defect: "Inspect bristles, brush head and handle for deformation, missing bristles or physical damage.",
                      verify: "Check bristle alignment, head condition and handle integrity."
                    },
                    transistor: {
                      defect: "Inspect the component body and leads for bending, displacement, cracks or physical damage.",
                      verify: "Check lead alignment and component body condition."
                    },
                    wood: {
                      defect: "Inspect the wood surface for scratches, cracks, holes, color variation or abnormal texture.",
                      verify: "Check surface appearance and structural condition."
                    },
                    zipper: {
                      defect: "Inspect zipper teeth, slider and fabric interface for broken, missing or misaligned components.",
                      verify: "Check tooth alignment and slider movement."
                    }
                  };

                  const advice = categoryAdvice[category] || {
                    defect: "Inspect the highlighted anomaly region against the applicable product quality specification.",
                    verify: "Perform manual QA verification before final disposition."
                  };

                  const titleCategory = category
                    ? category.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase())
                    : "Product";

                  const action =
                    result.status === "NORMAL"
                      ? "No automatic defect disposition is required. Complete the standard QA check before release."
                      : severity === "CRITICAL" && concentration === "WIDESPREAD"
                        ? `Quarantine the ${titleCategory.toLowerCase()} unit, stop release, and escalate for a broader QA/Manufacturing review because ${patchCount} high-scoring patches (${patchRatio.toFixed(2)}%) indicate a widespread anomaly.`
                        : severity === "CRITICAL"
                          ? `Quarantine the ${titleCategory.toLowerCase()} unit and escalate the localized anomaly to QA/Manufacturing for confirmation before any release decision.`
                          : severity === "HIGH" && concentration === "SIGNIFICANT"
                            ? `Place the ${titleCategory.toLowerCase()} unit on hold and inspect additional units from the same production interval because multiple high-scoring regions were detected.`
                            : severity === "HIGH"
                              ? `Place the ${titleCategory.toLowerCase()} unit on hold and perform targeted QA inspection of the highlighted anomaly region before release.`
                              : severity === "MODERATE" && concentration === "LOCALIZED"
                                ? `Hold the ${titleCategory.toLowerCase()} unit for targeted manual inspection of the localized heatmap region. Compare the area with a known-good reference before release.`
                                : `Perform manual verification of the ${titleCategory.toLowerCase()} unit and compare the highlighted region with a known-good reference before release.`;

                  const patchRecommendation =
                    concentration === "WIDESPREAD"
                      ? `The anomaly is distributed across a broad portion of the image. Inspect the overall product condition rather than only one defect location.`
                      : concentration === "SIGNIFICANT"
                        ? `Multiple high-scoring regions were detected. Inspect the main affected areas and nearby surfaces for related defects.`
                        : concentration === "LOCALIZED"
                          ? `The anomaly is concentrated in a smaller region. Focus manual inspection on the highlighted heatmap area.`
                          : `Only limited high-scoring patches were detected. Verify the highlighted region carefully against a known-good unit.`;


                  return (
                    <div className="action-list">
                      <div className={`action-item ${severity === "CRITICAL" ? "critical" : ""}`}>
                        <span className="action-icon">
                          {severity === "CRITICAL" ? "🚨" : severity === "HIGH" ? "⚠️" : "🔎"}
                        </span>
                        <div>
                          <strong>{severity} SEVERITY — {titleCategory}</strong>
                          <p>
                            Score {score.toFixed(2)} is {ratio.toFixed(2)}× the threshold, with {patchCount} high-scoring patches ({patchRatio.toFixed(2)}%). Concentration: {concentration.toLowerCase()}.
                          </p>
                        </div>
                      </div>

                      <div className="action-item">
                        <span className="action-icon">🏭</span>
                        <div>
                          <strong>Recommended Production Action</strong>
                          <p>{action}</p>
                        </div>
                      </div>

                      <div className="action-item">
                        <span className="action-icon">🔎</span>
                        <div>
                          <strong>{titleCategory} — Defect Inspection</strong>
                          <p>{advice.defect}</p>
                        </div>
                      </div>

                      <div className="action-item">
                        <span className="action-icon">🧩</span>
                        <div>
                          <strong>Patch-Level Analysis</strong>
                          <p>{patchRecommendation} Detected {patchCount} high-scoring patches out of {result.patch_count || 0} total patches.</p>
                        </div>
                      </div>

                      <div className="action-item">
                        <span className="action-icon">👨‍🔧</span>
                        <div>
                          <strong>Manual Verification</strong>
                          <p>{advice.verify}</p>
                        </div>
                      </div>

                      <div className="action-item">
                        <span className="action-icon">🔬</span>
                        <div>
                          <strong>Batch Investigation</strong>
                          <p>
                            {severity === "CRITICAL" || severity === "HIGH"
                              ? "Inspect additional units from the same batch or production interval for similar anomalies."
                              : "Sample additional units from the same batch if the defect is confirmed."
                            }
                          </p>
                        </div>
                      </div>

                      <div className="action-item">
                        <span className="action-icon">📋</span>
                        <div>
                          <strong>Quality Record</strong>
                          <p>
                            Record the category, anomaly score, threshold, severity and defect heatmap for traceability.
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>

              {result.heatmap && (
                <div className="heatmap-result">
                  <div className="heatmap-title">DEFECT LOCALIZATION</div>
                  <img
                    src={`http://127.0.0.1:8000${result.heatmap}`}
                    alt="Defect heatmap"
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Analytics() {
  return (
    <div className="page">
      <PageIntro
        eyebrow="ANALYTICS"
        title="Model evaluation"
        description="Measured performance generated from the SentinelVision evaluation pipeline."
      />

      <div className="analytics-list">
        {charts.map((chart) => (
          <ChartPanel
            key={chart.file}
            title={chart.title}
            description={chart.description}
            file={chart.file}
            full
          />
        ))}
      </div>
    </div>
  );
}

function Results({ navigate }) {
  const [data, setData] = useState([]);
  const [selected, setSelected] = useState("bottle");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadEvaluation = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/evaluation");

      if (!response.ok) {
        throw new Error("Unable to load evaluation data");
      }

      const json = await response.json();
      setData(json.categories || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEvaluation();
  }, []);

  const current = data.find((item) => item.category === selected);

  const mean = (key) =>
    data.length
      ? (data.reduce((sum, item) => sum + Number(item[key] || 0), 0) / data.length).toFixed(2)
      : "—";

  return (
    <div className="page">
      <PageIntro
        eyebrow="EVALUATION RESULTS"
        title="Measured project results"
        description="Interactive evaluation results from the current SentinelVision PatchCore pipeline across all 15 MVTec AD categories."
      />

      {loading && (
        <div className="glass evaluation-loading">
          Loading evaluation results...
        </div>
      )}

      {error && (
        <div className="glass evaluation-error">
          {error}
        </div>
      )}

      {!loading && !error && (
        <>
          <div className="result-grid">
            <ResultCard
              label="Mean AUROC"
              value={`${mean("image_auroc")}%`}
              description="Image-level detection"
            />

            <ResultCard
              label="Mean Average Precision"
              value={`${mean("average_precision")}%`}
              description="Image-level precision"
            />

            <ResultCard
              label="Mean Pixel AUROC"
              value={`${mean("pixel_auroc")}%`}
              description="Defect localization"
            />
          </div>

          <section className="evaluation-selector glass">
            <div>
              <span className="eyebrow">CATEGORY INSPECTION</span>
              <h2>Category performance</h2>
              <p>Select a category to inspect its measured evaluation metrics.</p>
            </div>

            <select
              value={selected}
              onChange={(event) => setSelected(event.target.value)}
            >
              {data.map((item) => (
                <option key={item.category} value={item.category}>
                  {item.category.replace("_", " ").toUpperCase()}
                </option>
              ))}
            </select>
          </section>

          {current && (
            <section className="selected-evaluation glass">
              <div className="selected-evaluation-header">
                <div>
                  <span className="eyebrow">SELECTED CATEGORY</span>
                  <h2>{current.category.replace("_", " ").toUpperCase()}</h2>
                </div>

                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => navigate("inspection")}
                >
                  <Activity size={16} />
                  Inspect Image
                </button>
              </div>

              <div className="evaluation-stat-grid">
                <div>
                  <span>Image AUROC</span>
                  <strong>{current.image_auroc}%</strong>
                </div>

                <div>
                  <span>Average Precision</span>
                  <strong>{current.average_precision}%</strong>
                </div>

                <div>
                  <span>Pixel AUROC</span>
                  <strong>{current.pixel_auroc}%</strong>
                </div>

                <div>
                  <span>Test Samples</span>
                  <strong>{current.test_samples}</strong>
                </div>

                <div>
                  <span>Threshold</span>
                  <strong>{current.threshold}</strong>
                </div>
              </div>
            </section>
          )}

          {current && (
            <section className="selected-category-chart glass">
              <div className="section-title">
                <div>
                  <span className="eyebrow">CATEGORY VISUALIZATION</span>
                  <h2>{current.category.replace("_", " ").toUpperCase()} performance</h2>
                </div>
                <BarChart3 size={22} />
              </div>

              <div className="metric-bars">
                <div className="metric-bar-row">
                  <div className="metric-bar-label">
                    <span>Image AUROC</span>
                    <strong>{current.image_auroc}%</strong>
                  </div>
                  <div className="metric-bar-track">
                    <div className="metric-bar-fill" style={{ width: `${current.image_auroc}%` }} />
                  </div>
                </div>

                <div className="metric-bar-row">
                  <div className="metric-bar-label">
                    <span>Average Precision</span>
                    <strong>{current.average_precision}%</strong>
                  </div>
                  <div className="metric-bar-track">
                    <div className="metric-bar-fill" style={{ width: `${current.average_precision}%` }} />
                  </div>
                </div>

                <div className="metric-bar-row">
                  <div className="metric-bar-label">
                    <span>Pixel AUROC</span>
                    <strong>{current.pixel_auroc}%</strong>
                  </div>
                  <div className="metric-bar-track">
                    <div className="metric-bar-fill" style={{ width: `${current.pixel_auroc}%` }} />
                  </div>
                </div>
              </div>
            </section>
          )}

          <section className="evaluation-table glass">
            <div className="section-title">
              <div>
                <span className="eyebrow">ALL CATEGORIES</span>
                <h2>Evaluation matrix</h2>
              </div>
              <Database size={22} />
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Image AUROC</th>
                    <th>Average Precision</th>
                    <th>Pixel AUROC</th>
                    <th>Samples</th>
                  </tr>
                </thead>

                <tbody>
                  {data.map((item) => (
                    <tr
                      key={item.category}
                      className={item.category === selected ? "selected-row" : ""}
                      onClick={() => setSelected(item.category)}
                    >
                      <td>{item.category.replace("_", " ")}</td>
                      <td>{item.image_auroc}%</td>
                      <td>{item.average_precision}%</td>
                      <td>{item.pixel_auroc}%</td>
                      <td>{item.test_samples}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {current && (
            <section className="selected-localization glass">
              <div className="section-title">
                <div>
                  <span className="eyebrow">SELECTED CATEGORY</span>
                  <h2>{current.category.replace("_", " ").toUpperCase()} defect localization</h2>
                  <p>Generated anomaly heatmap for the selected MVTec AD category.</p>
                </div>
                <Target size={22} />
              </div>

              <div className="selected-localization-content">
                <div className="selected-localization-image">
                  <img
                    src={`/heatmaps/${heatmaps.find((item) => item[0].toLowerCase().replace(" ", "_") === current.category)?.[1] || ""}`}
                    alt={`${current.category} defect localization`}
                  />
                </div>

                <div className="selected-localization-info">
                  <div>
                    <span>Category</span>
                    <strong>{current.category.replace("_", " ")}</strong>
                  </div>
                  <div>
                    <span>Pixel AUROC</span>
                    <strong>{current.pixel_auroc}%</strong>
                  </div>
                  <div>
                    <span>Defect Sample</span>
                    <strong>{heatmaps.find((item) => item[0].toLowerCase().replace(" ", "_") === current.category)?.[2] || "Generated heatmap"}</strong>
                  </div>
                  <div>
                    <span>Status</span>
                    <strong>Localization Available</strong>
                  </div>
                </div>
              </div>
            </section>
          )}

          <section className="evaluation-charts">
            <ChartPanel
              title="Image-Level AUROC"
              description="Measured anomaly detection performance across all evaluated categories."
              file="/results/01_auroc.png"
            />

            <ChartPanel
              title="Average Precision"
              description="Measured image-level precision across the MVTec AD categories."
              file="/results/02_average_precision.png"
            />

            <ChartPanel
              title="Localization Performance"
              description="Measured pixel-level anomaly localization performance."
              file="/results/04_localization.png"
            />

            <ChartPanel
              title="Overall Performance"
              description="Combined SentinelVision evaluation dashboard."
              file="/results/SentinelVision_results_dashboard.png"
              large
            />
          </section>

          <div className="results-footer">
            <div>
              <strong>Inspect generated defect localization</strong>
              <span>Browse the actual heatmaps produced for the evaluated categories.</span>
            </div>

            <button
              type="button"
              className="secondary-button"
              onClick={() => navigate("heatmaps")}
            >
              Open Heatmaps
              <ChevronRight size={16} />
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function HeatmapExplorer({ selected, setSelected }) {
  return (
    <div className="page">
      <PageIntro
        eyebrow="DEFECT LOCALIZATION"
        title="Heatmap explorer"
        description="Browse the generated anomaly localization outputs from the 15 evaluated MVTec AD categories."
      />

      <section className="heatmap-layout">
        <div className="category-panel glass">
          <div className="category-panel-header">
            <div>
              <span>CATEGORIES</span>
              <strong>15 evaluated samples</strong>
            </div>
          </div>

          <div className="category-list">
            {heatmaps.map((item) => (
              <button
                type="button"
                key={item[0]}
                className={`category-button ${
                  selected[0] === item[0] ? "selected" : ""
                }`}
                onClick={() => setSelected(item)}
              >
                <div className="category-index">
                  {String(heatmaps.indexOf(item) + 1).padStart(2, "0")}
                </div>

                <div className="category-text">
                  <strong>{item[0]}</strong>
                  <span>{item[2]}</span>
                </div>

                <ChevronRight size={15} />
              </button>
            ))}
          </div>
        </div>

        <div className="heatmap-panel glass">
          <div className="heatmap-header">
            <div>
              <div className="eyebrow">
                <span />
                SELECTED SAMPLE
              </div>

              <h2>{selected[0]}</h2>
              <p>Defect type: {selected[2]}</p>
            </div>

            <div className="sample-status">
              <span />
              TEST SAMPLE
            </div>
          </div>

          <div className="heatmap-image-container">
            <img
              src={`/heatmaps/${selected[1]}`}
              alt={`${selected[0]} ${selected[2]} heatmap`}
            />
          </div>

          <div className="heatmap-meta">
            <Meta label="Category" value={selected[0]} />
            <Meta label="Defect Type" value={selected[2]} />
            <Meta label="Output" value="Anomaly heatmap" />
          </div>
        </div>
      </section>
    </div>
  );
}

function ImageAnalysisCharts({ result }) {
  const score = Number(result.score || 0);
  const threshold = Number(result.threshold || 0);
  const patches = Array.isArray(result.patch_scores) ? result.patch_scores : [];
  const high = Number(result.high_score_patch_count || 0);
  const total = Number(result.patch_count || patches.length || 0);
  const ratio = total ? (high / total) * 100 : 0;
  const maxValue = Math.max(score, threshold, 1);
  const bins = [
    {
      label: "Below 1×",
      count: patches.filter((v) => v < threshold).length,
    },
    {
      label: "1×–1.25×",
      count: patches.filter((v) => v >= threshold && v < threshold * 1.25).length,
    },
    {
      label: "1.25×–2×",
      count: patches.filter((v) => v >= threshold * 1.25 && v < threshold * 2).length,
    },
    {
      label: "2×–3×",
      count: patches.filter((v) => v >= threshold * 2 && v < threshold * 3).length,
    },
    {
      label: ">3×",
      count: patches.filter((v) => v >= threshold * 3).length,
    },
  ];

  const maxBin = Math.max(...bins.map((b) => b.count), 1);

  return (
    <div className="image-analysis">
      <div className="image-analysis-header">
        <div>
          <span className="plan-eyebrow">IMAGE-SPECIFIC ANALYTICS</span>
          <h3>Inspection Analytics</h3>
        </div>
        <span className="image-analysis-badge">{total} PATCHES ANALYZED</span>
      </div>

      <div className="image-chart-grid">
        <div className="image-chart-card">
          <div className="image-chart-title">
            <strong>Anomaly Score vs Threshold</strong>
            <span>Image-level decision boundary</span>
          </div>

          <div className="score-bars">
            <div className="score-row">
              <span>Score</span>
              <div className="score-track">
                <div className="score-fill score-anomaly" style={{ width: `${Math.min((score / maxValue) * 100, 100)}%` }} />
              </div>
              <strong>{score.toFixed(2)}</strong>
            </div>

            <div className="score-row">
              <span>Threshold</span>
              <div className="score-track">
                <div className="score-fill score-threshold" style={{ width: `${Math.min((threshold / maxValue) * 100, 100)}%` }} />
              </div>
              <strong>{threshold.toFixed(2)}</strong>
            </div>
          </div>

          <div className="chart-footnote">
            Score ratio: <strong>{threshold ? (score / threshold).toFixed(2) : "0.00"}×</strong>
          </div>
        </div>

        <div className="image-chart-card">
          <div className="image-chart-title">
            <strong>Patch Score Distribution</strong>
            <span>Patch scores relative to the anomaly threshold</span>
          </div>

          <div className="patch-bars">
            {bins.map((bin, index) => (
              <div className="patch-bar-wrap" key={index}>
                <div
                  className="patch-bar"
                  style={{ height: `${Math.max((bin.count / maxBin) * 100, bin.count ? 8 : 2)}%` }}
                  title={`${bin.count} patches`}
                />
                <span>{bin.label}</span>
              </div>
            ))}
          </div>

          <div className="chart-footnote">
            Threshold: <strong>{threshold.toFixed(2)}</strong> · {patches.length} patches analyzed
          </div>
        </div>

        <div className="image-chart-card">
          <div className="image-chart-title">
            <strong>Patch Concentration</strong>
            <span>High-scoring regions</span>
          </div>

          <div className="concentration-visual">
            <div className="concentration-ring" style={{ "--ratio": ratio }}>
              <div>
                <strong>{ratio.toFixed(1)}%</strong>
                <span>high-score</span>
              </div>
            </div>

            <div className="concentration-stats">
              <div><span>High-scoring</span><strong>{high}</strong></div>
              <div><span>Remaining</span><strong>{Math.max(total - high, 0)}</strong></div>
              <div><span>Severity</span><strong>{result.severity || "—"}</strong></div>
              <div><span>Pattern</span><strong>{result.anomaly_concentration || "—"}</strong></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ icon, label, value, detail }) {
  return (
    <div className="metric-card glass">
      <div className="metric-icon">{icon}</div>
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function ResultCard({ label, value, description }) {
  return (
    <div className="result-card glass">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{description}</small>
    </div>
  );
}

function ChartPanel({ title, description, file, large, full }) {
  return (
    <section
      className={`chart-panel glass ${large ? "large" : ""} ${
        full ? "full" : ""
      }`}
    >
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>

        <Activity size={18} />
      </div>

      <div className="chart-image">
        <img src={file} alt={title} />
      </div>
    </section>
  );
}

function MethodCard({ number, title, text }) {
  return (
    <div className="method-card glass">
      <span>{number}</span>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

function SectionHeading({ eyebrow, title, action, onClick }) {
  return (
    <div className="section-heading">
      <div>
        <div className="eyebrow">
          <span />
          {eyebrow}
        </div>
        <h2>{title}</h2>
      </div>

      {action && (
        <button type="button" className="link-button" onClick={onClick}>
          {action}
          <ChevronRight size={15} />
        </button>
      )}
    </div>
  );
}

function PageIntro({ eyebrow, title, description }) {
  return (
    <section className="page-intro">
      <div className="eyebrow">
        <span />
        {eyebrow}
      </div>
      <h1>{title}</h1>
      <p>{description}</p>
    </section>
  );
}

function Meta({ label, value }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;

