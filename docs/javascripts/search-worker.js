importScripts('https://cdn.jsdelivr.net/npm/algoliasearch@5.18.0/dist/algoliasearch.umd.min.js')

const SETUP = 0
const READY = 1
const QUERY = 2
const RESULT = 3

// same constants are present in docs/.hooks/algolia.py.
// If you change them here, change them there as well.
const appID = 'KPPUDTIAVX';
const apiKey = '1fc841595212a2c3afe8c24dd4cb8790';
const indexName = 'pydantic-ai-docs';

const client = algoliasearch.algoliasearch(appID, apiKey);

self.onmessage = async (event) => {
  if (event.data.type === SETUP) {
    self.postMessage({ type: READY });
  } else if (event.data.type === QUERY) {

    const query = event.data.data

    if (query === '') {
      self.postMessage({
        type: RESULT, data: {
          items: []
        }
      });
      return
    }

    const { results } = await client.search({
      requests: [
        {
          indexName,
          query,
        },
      ],
    });

    const hits = results[0].hits

    // make navigation work with preview deployments
    const stripDocsPathName = !(new URL(self.location.href).pathname.startsWith('/docs'));

    const mappedGroupedResults = hits.reduce((acc, hit) => {
      if (!acc[hit.pageID]) {
        acc[hit.pageID] = []
      }
      acc[hit.pageID].push({
        score: 1,
        terms: {},
        location: stripDocsPathName ? hit.abs_url.replace('/docs', '') : hit.abs_url,
        title: hit.title,
        text: hit._highlightResult.content.value,

      })
      return acc
    }, {})




    self.postMessage({
      type: RESULT, data: {
        items: Object.values(mappedGroupedResults)
      }
    });
  }
};
