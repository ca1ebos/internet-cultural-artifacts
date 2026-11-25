'use strict';

const { models, cases, answers, metrics, breakdown, categories, highlights, run } = window.CARLB;
const byId = id => document.getElementById(id);
const modelIds = Object.keys(models);
const categoryOptions = Object.keys(categories).map(name => [name, name]);
const casesById = new Map(cases.map(item => [item.id, item]));

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function setOptions(select, options) {
  select.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
}

function renderFindings() {
  byId('case-count').textContent = cases.length;
  byId('model-count').textContent = modelIds.length;
  byId('reply-count').textContent = run.completed;
  const otherTotal = (model, key) => modelIds.filter(id => id !== model).reduce((sum, id) => sum + metrics[id][key], 0);
  const cards = [
    ['The base ran long', `It reached the ${run.max_tokens}-token reply cap on ${metrics.base.token_caps}/${metrics.base.responses} prompts. The five adapters combined hit the cap ${otherTotal('base', 'token_caps')} times.`],
    ['LinkedIn turned prompts into posts', `It opened with the full prompt in ${metrics.linkedin.echoes}/${metrics.linkedin.responses} replies and used hashtags in ${metrics.linkedin.hashtags}. The other five models combined echoed ${otherTotal('linkedin', 'echoes')} times.`],
    ['Quora leaned on experience', `Selected experience phrases appeared in ${metrics.quora.experience}/${metrics.quora.responses} Quora replies, compared with ${otherTotal('quora', 'experience')} occurrence across the other five models combined.`]
  ];
  byId('findings').replaceChildren(...cards.map(([title, body]) => {
    const card = node('article', null, 'finding');
    card.append(node('h3', title), node('p', body));
    return card;
  }));
  const { temperature, repeat_penalty: penalty, top_k: k, top_p: p } = run.sampling;
  byId('settings').textContent = `Six ${run.quantization} exports on llama.cpp ${run.llama_cpp}, with a fresh chat for each prompt, ${run.context.toLocaleString()} context tokens, a ${run.max_tokens}-token reply cap, temperature ${temperature}, top-k ${k}, top-p ${p}, and a ${penalty} repetition penalty.`;
}

function renderScoreboard() {
  const category = byId('score-category').value;
  const selected = category ? breakdown[category] : metrics;
  byId('score-rows').replaceChildren(...modelIds.map(model => {
    const info = selected[model];
    const row = node('tr');
    const name = node('th', models[model]);
    name.scope = 'row';
    const length = node('td', null, 'bar-cell');
    const track = node('div', null, 'bar-track');
    const bar = node('progress');
    bar.value = info.median_tokens;
    bar.max = run.max_tokens;
    bar.setAttribute('aria-hidden', 'true');
    track.append(bar, node('span', info.median_tokens));
    length.append(track);
    const rate = value => `${value}/${info.responses}`;
    row.append(name, length, node('td', rate(info.token_caps)), node('td', rate(info.echoes)),
      node('td', rate(info.experience)), node('td', rate(info.hashtags)),
      node('td', `${(info.mean_repetition * 100).toFixed(2)}%`));
    return row;
  }));
}

function renderHighlights() {
  const model = byId('highlight-model').value;
  const tag = byId('highlight-tag').value;
  const selected = highlights.filter(item => (!model || item.model === model) && (!tag || item.tag === tag));
  const cards = selected.map(item => {
    const card = node('article', null, 'card');
    const link = node('a', `Compare ${item.id} →`, 'case-link');
    link.href = `#case=${item.id}`;
    card.append(node('p', `${models[item.model]} · ${item.tag}`, 'meta'),
      node('h3', item.title), node('blockquote', item.excerpt), node('p', item.note, 'card-note'), link);
    return card;
  });
  byId('highlight-count').textContent = `${selected.length} highlights`;
  byId('highlight-cards').replaceChildren(...(cards.length ? cards : [node('p', 'No highlights match these filters.', 'empty')]));
}

function renderReply(caseId) {
  const item = casesById.get(caseId);
  if (!item) {
    byId('case-heading').replaceChildren(node('p', 'No prompts match this search.', 'empty'));
    byId('case-replies').replaceChildren();
    return;
  }
  byId('case-heading').replaceChildren(node('p', `${item.id} · ${item.category}`, 'meta'), node('h3', item.prompt));
  byId('case-replies').replaceChildren(...modelIds.map(model => {
    const reply = answers[caseId][model];
    const card = node('article', null, 'reply');
    const details = node('details');
    const note = reply.info.finish_reason === 'length' ? ' · reached cap' : '';
    const response = node('div', null, 'response');
    // The build parses Markdown from replies with raw HTML and images disabled.
    response.innerHTML = reply.html;
    details.append(node('summary', 'Read full reply'), response);
    card.append(node('h4', models[model]), node('p', `${reply.info.completion_tokens} tokens${note}`, 'meta'), details);
    return card;
  }));
}

function renderCaseList(preferred = byId('case-select').value) {
  const category = byId('reply-category').value;
  const search = byId('case-search').value.trim().toLowerCase();
  const found = cases.filter(item => (!category || item.category === category) &&
    (!search || `${item.id} ${item.prompt}`.toLowerCase().includes(search)));
  const select = byId('case-select');
  setOptions(select, found.map(item => [item.id, `${item.id} · ${item.prompt}`]));
  select.disabled = found.length === 0;
  if (found.some(item => item.id === preferred)) select.value = preferred;
  renderReply(select.value);
}

function followCaseLink() {
  const match = /^#case=(CARLB-\d+)$/.exec(window.location.hash);
  if (!match || !casesById.has(match[1])) return;
  byId('reply-category').value = '';
  byId('case-search').value = '';
  renderCaseList(match[1]);
  byId('replies').scrollIntoView({ block: 'start' });
  byId('case-select').focus({ preventScroll: true });
}

renderFindings();
setOptions(byId('score-category'), [['', 'All categories'], ...categoryOptions]);
setOptions(byId('reply-category'), [['', 'All categories'], ...categoryOptions]);
setOptions(byId('highlight-model'), [['', 'All models'], ...Object.entries(models)]);
setOptions(byId('highlight-tag'), [['', 'All themes'], ...Array.from(new Set(highlights.map(item => item.tag)), tag => [tag, tag])]);
byId('score-category').addEventListener('change', renderScoreboard);
byId('highlight-model').addEventListener('change', renderHighlights);
byId('highlight-tag').addEventListener('change', renderHighlights);
byId('reply-category').addEventListener('change', () => renderCaseList());
byId('case-search').addEventListener('input', () => renderCaseList());
byId('case-select').addEventListener('change', event => renderReply(event.target.value));
window.addEventListener('hashchange', followCaseLink);
renderScoreboard();
renderHighlights();
renderCaseList('CARLB-037');
followCaseLink();
