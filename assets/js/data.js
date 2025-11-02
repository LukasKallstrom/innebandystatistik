let datasetPromise;

export function loadGoalieDataset() {
  if (!datasetPromise) {
    datasetPromise = fetch('assets/data/goalies.json')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load goalie dataset: ${response.status}`);
        }
        return response.json();
      })
      .catch((error) => {
        console.error('Unable to load goalie dataset', error);
        throw error;
      });
  }
  return datasetPromise;
}

export async function getSummaryData() {
  const data = await loadGoalieDataset();
  return data.summary;
}

export async function getGoalieMap() {
  const data = await loadGoalieDataset();
  return data.goalieToTeam;
}

export async function getTimelineBundle() {
  const data = await loadGoalieDataset();
  return data.timeline;
}

export async function getTraceOrder() {
  const data = await loadGoalieDataset();
  return data.traceOrder;
}
