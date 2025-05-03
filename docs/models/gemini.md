# Gemini

PydanticAI supports Google's Gemini models through two different APIs:

- Generative Language API (`generativelanguage.googleapis.com`)
- Vertex AI API (`*-aiplatform.googleapis.com`)

## Install

To use `GeminiModel` models, you need to either install `pydantic-ai`, or install
`pydantic-ai-slim` with the `gemini` optional group:

```bash
pip/uv-add "pydantic-ai-slim[gemini]"
```

## Gemini via Generative Language API

To use Google Generative Language API, you need to set the `GEMINI_API_KEY` environment variable.

### Configuration

`GeminiModel` lets you use Google's Gemini models through their [Generative Language API](https://ai.google.dev/api/all-methods), `generativelanguage.googleapis.com`.

`GeminiModelName` contains a list of available Gemini models that can be used through this interface.

To use `GeminiModel`, go to [aistudio.google.com](https://aistudio.google.com/apikey) and select "Create API key".

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export GEMINI_API_KEY=your-api-key
```

You can then use `GeminiModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-2.0-flash')
...
```

!!! note
    The `google-gla` provider prefix represents the [Google **G**enerative **L**anguage **A**PI](https://ai.google.dev/api/all-methods) for `GeminiModel`s.
    `google-vertex` is used with [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).

Or initialize the model directly with just the model name and provider:

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel('gemini-2.0-flash', provider='google-gla')
agent = Agent(model)
...
```

### `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google import GoogleProvider

model = GeminiModel(
    'gemini-2.0-flash', provider=GoogleProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

You can also customize the [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] with a
custom `client`:

```python
from google import genai

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google import GoogleProvider

custom_client = genai.Client(api_key='your-api-key', http_options={'timeout': 30})
model = GeminiModel(
    'gemini-2.0-flash',
    provider=GoogleProvider(client=custom_client),
)
agent = Agent(model)
...
```

## Gemini via VertexAI

If you are an enterprise user, you should use the `google-vertex` provider with `GeminiModel`
which uses the `*-aiplatform.googleapis.com` API.

`GeminiModelName` contains a list of available Gemini models that can be used through this interface.

### Configuration

This interface has a number of advantages over `generativelanguage.googleapis.com` documented above:

1. The VertexAI API comes with more enterprise readiness guarantees.
2. You can [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput) with VertexAI to guarantee capacity.
3. If you're running PydanticAI inside GCP, you don't need to set up authentication, it should "just work".
4. You can decide which region to use, which might be important from a regulatory perspective, and might improve latency.

The big disadvantage is that for local development you may need to create and configure a "service account", which can be challenging to get right.

Whichever way you authenticate, you'll need to have VertexAI enabled in your GCP account.

### Application default credentials

Luckily if you're running PydanticAI inside GCP, or you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you should be able to use `VertexAIModel` without any additional setup.

To use `VertexAIModel`, with [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials) configured (e.g. with `gcloud`), you can simply use:

```python {title="gemini_application_default_credentials.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel('gemini-2.0-flash', provider='google-vertex')
agent = Agent(model)
...
```

Internally this uses [`google.auth.default()`](https://google-auth.readthedocs.io/en/master/reference/google.auth.html) from the `google-auth` package to obtain credentials.

!!! note "Won't fail until `agent.run()`"
    Because `google.auth.default()` requires network requests and can be slow, it's not run until you call `agent.run()`.

You may also need to pass the `project_id` argument to `GoogleProvider` if application default credentials don't set a project, if you pass `project_id` and it conflicts with the project set by application default credentials, an error is raised.

### Service account

If instead of application default credentials, you want to authenticate with a service account, you'll need to create a service account, add it to your GCP project (note: this step is necessary even if you created the service account within the project), give that service account the "Vertex AI Service Agent" role, and download the service account JSON file.

Once you have the JSON file, you can use it thus:

```python {title="gemini_service_account_file.py" test="skip"}
from google.oauth2 import service_account

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google import GoogleProvider

credentials = service_account.Credentials.from_service_account_file(
    'path/to/service-account.json'
)

model = GeminiModel(
    'gemini-2.0-flash',
    provider=GoogleProvider(credentials=credentials),
)
agent = Agent(model)
...
```

Alternatively, if you already have the service account information in memory, you can pass it as a dictionary:

```python
from google.oauth2 import service_account

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google import GoogleProvider

credentials = service_account.Credentials.from_service_account_info(
    {
        'type': 'service_account',
        'project_id': 'my-project-id',
        'client_email': 'my-client-email@mail.com',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'private_key': (
            '-----BEGIN PRIVATE KEY-----\n'
            'MIICdgIBADANBgkqhkiG9w0BAQEFAASCAmAwggJcAgEAAoGBAMFrZYX4gZ20qv88\n'
            'jD0QCswXgcxgP7Ta06G47QEFprDVcv4WMUBDJVAKofzVcYyhsasWsOSxcpA8LIi9\n'
            '/VS2Otf8CmIK6nPBCD17Qgt8/IQYXOS4U2EBh0yjo0HQ4vFpkqium4lLWxrAZohA\n'
            '8r82clV08iLRUW3J+xvN23iPHyVDAgMBAAECgYBScRJe3iNxMvbHv+kOhe30O/jJ\n'
            'QiUlUzhtcEMk8mGwceqHvrHTcEtRKJcPC3NQvALcp9lSQQhRzjQ1PLXkC6BcfKFd\n'
            '03q5tVPmJiqsHbSyUyHWzdlHP42xWpl/RmX/DfRKGhPOvufZpSTzkmKWtN+7osHu\n'
            '7eiMpg2EDswCvOgf0QJBAPXLYwHbZLaM2KEMDgJSse5ZTE/0VMf+5vSTGUmHkr9c\n'
            'Wx2G1i258kc/JgsXInPbq4BnK9hd0Xj2T5cmEmQtm4UCQQDJc02DFnPnjPnnDUwg\n'
            'BPhrCyW+rnBGUVjehveu4XgbGx7l3wsbORTaKdCX3HIKUupgfFwFcDlMUzUy6fPO\n'
            'IuQnAkA8FhVE/fIX4kSO0hiWnsqafr/2B7+2CG1DOraC0B6ioxwvEqhHE17T5e8R\n'
            '5PzqH7hEMnR4dy7fCC+avpbeYHvVAkA5W58iR+5Qa49r/hlCtKeWsuHYXQqSuu62\n'
            'zW8QWBo+fYZapRsgcSxCwc0msBm4XstlFYON+NoXpUlsabiFZOHZAkEA8Ffq3xoU\n'
            'y0eYGy3MEzxx96F+tkl59lfkwHKWchWZJ95vAKWJaHx9WFxSWiJofbRna8Iim6pY\n'
            'BootYWyTCfjjwA==\n'
            '-----END PRIVATE KEY-----\n'
        )
    }
)
model = GeminiModel(
    'gemini-2.0-flash',
    provider=GoogleProvider(credentials=credentials),
)
agent = Agent(model)
...
```

### Customize location

Whichever way you authenticate, you can specify which location requests will be sent to via the `location` argument.

Using a location close to your application can improve latency and might be important from a regulatory perspective.

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google import GoogleProvider

model = GeminiModel(
    'gemini-2.0-flash', provider=GoogleProvider(location='asia-east1')
)
agent = Agent(model)
...
```

You can also customize the [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] with a
custom `genai.Client`:

```python
from google import genai

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google import GoogleProvider

custom_client = genai.Client(api_key='your-api-key')
model = GeminiModel(
    'gemini-2.0-flash',
    provider=GoogleProvider(location='asia-east1', client=custom_client),
)
agent = Agent(model)
...
```
