'use strict';
const $ = id => document.getElementById(id);
let runs = [], selected = [], current = null, requestVersion = 0;
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
async function get(url) { const r = await fetch(url); const data = await r.json(); if (!r.ok) throw new Error(data.error || 'Request failed'); return data; }
function option(value, text) { const node = document.createElement('option'); node.value = value; node.textContent = text; return node; }
function filterSamples() {
  const previous = $('sample').value, query = $('search').value.toLowerCase(), mode = $('filter').value;
  const samples = runs[Number($('run').value)]?.samples || [];
  selected = samples.filter(s => (s.id.toLowerCase().includes(query) || s.question.toLowerCase().includes(query)) && (mode === 'all' || mode === 'correct' && s.correct === true || mode === 'incorrect' && s.correct === false || mode === 'unscored' && s.correct == null || mode === 'length' && s.finish_reason === 'length'));
  $('sample').replaceChildren(...selected.map(s => option(s.index, `Sample ${s.id}`)));
  if (selected.some(s => String(s.index) === previous)) $('sample').value = previous;
  const correct = samples.filter(s => s.correct === true).length, scored = samples.filter(s => typeof s.correct === 'boolean').length;
  $('totals').textContent = `${samples.length} samples · ${scored ? `${correct}/${scored} correct` : 'Unscored'}`;
  loadSample();
}
function renderOutput() {
  if (!current) return;
  const raw = $('raw-mode').checked, literal = $('literal-breaks').checked;
  // JSON parsing already restores ordinary escaped line breaks; this optional view handles double escaping.
  let text = current.response;
  if (literal) text = text.replace(/\\r\\n|\\n/g, '\n');
  $('response').hidden = raw || literal;
  $('raw-response').hidden = !raw && !literal;
  $('raw-response').textContent = text || 'No generated content was saved.';
  $('response').innerHTML = current.html.response || '<p>No generated content was saved.</p>';
}
async function loadSample() {
  const version = ++requestVersion;
  current = null;
  $('workspace').hidden = true;
  $('empty').hidden = selected.length !== 0;
  const index = Number($('sample').value), position = selected.findIndex(s => s.index === index);
  $('position').textContent = selected.length ? `${position + 1} / ${selected.length}` : '0 / 0';
  $('previous').disabled = position <= 0; $('next').disabled = position < 0 || position >= selected.length - 1;
  if (!selected.length) return;
  try {
    const query = `run=${$('run').value}&index=${index}`;
    const data = await get(`/api/sample?${query}`);
    if (version !== requestVersion) return;
    current = data; showError('');
    $('sample-id').textContent = `# ${data.id}`;
    $('category').textContent = data.category;
    $('model').textContent = data.model || 'Model response';
    $('question').innerHTML = data.html.question;
    $('reference').innerHTML = data.html.reference || '<span>No reference answer available.</span>';
    $('reference-reasoning-block').hidden = !data.reference_reasoning;
    $('reference-reasoning').innerHTML = data.html.reference_reasoning;
    $('model-reasoning-block').hidden = !data.reasoning;
    $('model-reasoning').innerHTML = data.html.reasoning;
    $('prompt').textContent = data.prompt;
    $('image').src = `/api/image?${query}`;
    $('large-image').src = `/api/image?${query}`;
    $('badges').replaceChildren();
    const status = data.correct === true ? 'correct' : data.correct === false ? 'incorrect' : 'unscored';
    for (const [text, cls] of [[status, status], [data.finish_reason, data.finish_reason], [data.tokens == null ? '' : `${data.tokens} tokens`, ''], [data.prediction ? `Prediction: ${data.prediction}` : '', '']]) {
      if (!text) continue;
      const el = document.createElement('span'); el.className = `chip ${cls}`; el.textContent = text; $('badges').append(el);
    }
    renderOutput(); $('workspace').hidden = false;
  } catch (e) { if (version === requestVersion) showError(e.message); }
}
async function loadRuns() {
  try { const previous = $('run').value; runs = await get('/api/runs'); $('run').replaceChildren(...runs.map((r,i) => option(i,r.name))); if (previous !== '' && runs[Number(previous)]) $('run').value = previous; filterSamples(); }
  catch (e) { showError(e.message); }
}
$('run').onchange = filterSamples; $('search').oninput = filterSamples; $('filter').onchange = filterSamples; $('sample').onchange = loadSample;
for (const [id, step] of [['previous',-1],['next',1]]) $(id).onclick = () => { const p = selected.findIndex(s => s.index === Number($('sample').value)); if (selected[p+step]) { $('sample').value = selected[p+step].index; loadSample(); } };
$('raw-mode').onchange = renderOutput; $('literal-breaks').onchange = renderOutput;
$('copy').onclick = async () => { if (!current) return; try { await navigator.clipboard.writeText(current.response); $('copy').textContent = 'Copied'; setTimeout(() => $('copy').textContent = 'Copy output', 1500); } catch { showError('Clipboard unavailable. Select Plain text and copy the response manually.'); } };
$('zoom').onclick = () => $('image-dialog').showModal(); $('close-dialog').onclick = () => $('image-dialog').close();
$('image').onerror = () => showError('Unable to load this sample image. Check the image path saved in the JSONL file.');
$('reload').onclick = () => location.reload();
loadRuns();
