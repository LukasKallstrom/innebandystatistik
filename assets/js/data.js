let datasetPromise;

function buildDatasetUrl(cacheBust) {
  const url = new URL('assets/data/goalies.json', window.location.href);
  if (cacheBust) {
    url.searchParams.set('_', Date.now().toString());
  }
  return url.toString();
}

export function loadGoalieDataset(options = {}) {
  const { cacheBust = false } = options;
  if (!datasetPromise || cacheBust) {
    const fetchUrl = buildDatasetUrl(cacheBust);
    const fetchOptions = cacheBust ? { cache: 'no-store' } : undefined;
    datasetPromise = fetch(fetchUrl, fetchOptions)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load goalie dataset: ${response.status}`);
        }
        return response.json();
      })
      .catch((error) => {
        console.error('Unable to load goalie dataset', error);
        datasetPromise = undefined;
        throw error;
      });
  }
  return datasetPromise;
}

export async function getSummaryData(options = {}) {
  const data = await loadGoalieDataset(options);
  return data.summary;
}

export async function getGoalieMap(options = {}) {
  const data = await loadGoalieDataset(options);
  return data.goalieToTeam;
}

export async function getTimelineBundle(options = {}) {
  const data = await loadGoalieDataset(options);
  return data.timeline;
}

export async function getTraceOrder(options = {}) {
  const data = await loadGoalieDataset(options);
  return data.traceOrder;
}

export async function getUpdatedAt(options = {}) {
  const data = await loadGoalieDataset(options);
  return data.updatedAt ?? null;
}
