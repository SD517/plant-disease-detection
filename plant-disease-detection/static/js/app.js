(function () {
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), file = $("file"), cam = $("cam");
  const stage = $("stage"), photo = $("photo"), heat = $("heat");
  const empty = $("empty"), loading = $("loading"), err = $("error"), out = $("out");
  const blend = $("blend"), camControl = $("camControl");
  let photoUrl = null;

  const modelValue = () => {
    const r = document.querySelector('input[name="model"]:checked');
    if (r) return r.value;
    const h = document.querySelector('input[name="model"][type="hidden"]');
    return h ? h.value : "";
  };

  $("pick").addEventListener("click", (e) => { e.stopPropagation(); file.click(); });
  $("snap").addEventListener("click", (e) => { e.stopPropagation(); cam.click(); });
  drop.addEventListener("click", () => { if (!drop.classList.contains("has-image")) file.click(); });
  drop.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !drop.classList.contains("has-image")) { e.preventDefault(); file.click(); }
  });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) handle(e.dataTransfer.files[0]); });
  file.addEventListener("change", () => file.files[0] && handle(file.files[0]));
  cam.addEventListener("change", () => cam.files[0] && handle(cam.files[0]));
  blend.addEventListener("input", () => { heat.style.opacity = blend.value / 100; });

  // Re-run when the model is switched and an image is already loaded
  document.querySelectorAll('input[name="model"]').forEach((r) => r.addEventListener("change", () => {
    if (lastFile) analyse(lastFile);
  }));

  let lastFile = null;

  function show(which) {
    empty.hidden = which !== "empty"; loading.hidden = which !== "loading";
    err.hidden = which !== "error"; out.hidden = which !== "out";
  }

  function addChangeButton() {
    if ($("change")) return;
    const b = document.createElement("button");
    b.id = "change"; b.type = "button"; b.className = "btn ghost change"; b.textContent = "Try another photo";
    b.addEventListener("click", () => file.click());
    drop.insertAdjacentElement("afterend", b);
  }

  function handle(f) {
    if (!f.type.startsWith("image/")) { err.textContent = "Please choose an image file."; show("error"); return; }
    if (photoUrl) URL.revokeObjectURL(photoUrl);
    photoUrl = URL.createObjectURL(f);
    photo.src = photoUrl;
    heat.removeAttribute("src"); heat.style.opacity = blend.value / 100;
    stage.hidden = false; drop.classList.add("has-image"); camControl.hidden = true;
    addChangeButton();
    lastFile = f;
    analyse(f);
  }

  async function analyse(f) {
    show("loading");
    const fd = new FormData();
    fd.append("image", f);
    fd.append("model", modelValue());
    try {
      const res = await fetch("/api/predict", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");
      render(data);
    } catch (e) {
      err.textContent = e.message; show("error");
    }
  }

  function pct(x) { return (x * 100).toFixed(1) + "%"; }

  function render(d) {
    const p = d.prediction, bad = !p.healthy;
    const st = $("status");
    st.className = "status " + (bad ? "bad" : "ok");
    st.textContent = bad ? "Disease detected" : "Looks healthy";
    $("title").textContent = bad ? p.condition : "Healthy " + p.crop.toLowerCase() + " leaf";
    $("sub").textContent = bad ? p.crop + " leaf. Cause: " + p.cause.toLowerCase() + "." : p.crop + " leaf.";
    document.querySelector(".meter").className = "meter" + (bad ? " bad" : "");
    $("meterFill").style.width = (p.confidence * 100).toFixed(1) + "%";
    $("confText").textContent = pct(p.confidence);
    $("warn").hidden = !d.low_confidence;
    $("desc").textContent = p.description;
    $("adviceHead").textContent = bad ? "What you can do" : "Keeping it healthy";
    const ul = $("advice"); ul.innerHTML = "";
    p.advice.forEach((a) => { const li = document.createElement("li"); li.textContent = a; ul.appendChild(li); });
    const top = $("top"); top.innerHTML = "";
    d.top.forEach((t) => {
      const li = document.createElement("li");
      const a = document.createElement("span"); a.textContent = t.crop + ": " + t.condition;
      const b = document.createElement("span"); b.textContent = pct(t.confidence);
      li.append(a, b); top.appendChild(li);
    });
    $("meta").textContent = "Model: " + d.model + " | " + d.inference_ms + " ms";
    if (d.heatmap) { heat.src = d.heatmap; camControl.hidden = false; }
    show("out");
  }
})();
