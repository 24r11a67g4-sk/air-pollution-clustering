const state = { data: null, busy: false };

const byId = (id) => document.getElementById(id);

function showToast(message, isError = false) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.style.borderLeftColor = isError ? "#ef8a6a" : "#80d4b3";
  toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("visible"), 3200);
}

function formatNumber(value) {
  return Number(value).toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function classificationClass(label) {
  const value = label.toLowerCase();
  if (value.includes("severe")) return "severe";
  if (value.includes("high")) return "high";
  if (value.includes("moderate")) return "moderate";
  return "low";
}

function renderSummary(result) {
  const { summary, extremes } = result;
  byId("total-records").textContent = formatNumber(summary.total_records);
  byId("average-aqi").textContent = formatNumber(summary.average_aqi);
  byId("highest-aqi").textContent = formatNumber(summary.highest_aqi);
  byId("lowest-aqi").textContent = formatNumber(summary.lowest_aqi);
  byId("cluster-count").textContent = summary.number_of_clusters;
  byId("highest-city").textContent = extremes.most_polluted_city;
  byId("lowest-city").textContent = extremes.least_polluted_city;
  byId("most-polluted").textContent = extremes.most_polluted_city;
  byId("least-polluted").textContent = extremes.least_polluted_city;
  byId("most-polluted-detail").textContent = `Highest recorded AQI: ${formatNumber(extremes.highest_aqi)}.`;
  byId("least-polluted-detail").textContent = `Lowest recorded AQI: ${formatNumber(extremes.lowest_aqi)}.`;
  byId("row-count").textContent = `${summary.total_records} rows`;
  byId("filename-label").textContent = result.filename;
  byId("data-status").textContent = `${result.filename} loaded · ${summary.missing_values} missing values handled`;
  byId("model-status").textContent = `K-Means complete · k = ${summary.number_of_clusters}`;
}

function renderDataset(records) {
  const body = byId("dataset-body");
  body.innerHTML = records.map((record) => `
    <tr>
      <td>${record.City}</td><td>${formatNumber(record["PM2.5"])}</td><td>${formatNumber(record.PM10)}</td>
      <td>${formatNumber(record.NO2)}</td><td>${formatNumber(record.SO2)}</td><td>${formatNumber(record.CO)}</td>
      <td><strong>${formatNumber(record.AQI)}</strong></td>
      <td><span class="cluster-cell"><span class="cluster-swatch"></span>${record.Cluster}</span></td>
    </tr>`).join("");
}

function renderResults(statistics) {
  byId("results-body").innerHTML = statistics.map((stat) => `
    <tr>
      <td><span class="cluster-cell"><span class="cluster-swatch"></span>Cluster ${stat.cluster}</span></td>
      <td>${stat.records}</td><td><strong>${formatNumber(stat.average_aqi)}</strong></td>
      <td>${formatNumber(stat.average_pm25)}</td>
      <td><span class="classification ${classificationClass(stat.classification)}">${stat.classification}</span></td>
    </tr>`).join("");
}

function renderPlots(result) {
  const version = `?v=${Date.now()}`;
  Object.entries({
    "elbow-plot": result.plots.elbow,
    "aqi-plot": result.plots.aqi_chart,
    "pm25-plot": result.plots.pm25_chart,
    "clusters-plot": result.plots.clusters,
    "distribution-plot": result.plots.cluster_distribution,
  }).forEach(([id, path]) => {
    byId(id).src = `${path}${version}`;
  });
}

function renderResult(result) {
  state.data = result;
  renderSummary(result);
  renderDataset(result.records);
  renderResults(result.statistics);
  renderPlots(result);
}

async function runAnalysis({ file = null, scroll = true } = {}) {
  if (state.busy) return;
  state.busy = true;
  const button = byId("run-analysis");
  const originalText = button.innerHTML;
  button.disabled = true;
  button.innerHTML = "Running K-Means…";
  const formData = new FormData();
  formData.append("k", document.querySelector('input[name="clusters"]:checked').value);
  if (file) formData.append("file", file);

  try {
    const response = await fetch("/cluster", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Unable to run analysis.");
    renderResult(payload);
    showToast(file ? "New dataset analyzed successfully." : "Clustering analysis refreshed.");
    if (scroll) {
      document.querySelector("#results").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (error) {
    showToast(error.message, true);
  } finally {
    state.busy = false;
    button.disabled = false;
    button.innerHTML = originalText;
  }
}

function setupInteractions() {
  document.querySelectorAll('input[name="clusters"]').forEach((input) => {
    input.addEventListener("change", () => {
      document.querySelectorAll(".cluster-picker label").forEach((label) => label.classList.remove("selected"));
      input.closest("label").classList.add("selected");
    });
  });

  byId("run-analysis").addEventListener("click", () => runAnalysis());
  byId("upload-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const file = byId("csv-file").files[0];
    if (!file) {
      showToast("Choose a CSV file first.", true);
      return;
    }
    runAnalysis({ file });
  });
  byId("csv-file").addEventListener("change", (event) => {
    const file = event.target.files[0];
    byId("file-name").textContent = file ? file.name : "Choose CSV file";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupInteractions();
  runAnalysis({ scroll: false });
});