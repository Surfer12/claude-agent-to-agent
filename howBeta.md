How to use beta headers
To access beta features, include the anthropic-beta header in your API requests:


Copy
POST /v1/messages
Content-Type: application/json
X-API-Key: YOUR_API_KEY
anthropic-beta: BETA_FEATURE_NAME
When using the SDK, you can specify beta headers in the request options:


Python

TypeScript

cURL

Copy
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ],
    extra_headers={
        "anthropic-beta": "beta-feature-name"
    }
)
Beta features are experimental and may:

Have breaking changes without notice
Be deprecated or removed
Have different rate limits or pricing
Not be available in all regions
​
Multiple beta features
To use multiple beta features in a single request, include all feature names in the header separated by commas:


Copy
anthropic-beta: feature1,feature2,feature3
​
Version naming conventions
Beta feature names typically follow the pattern: feature-name-YYYY-MM-DD, where the date indicates when the beta version was released. Always use the exact beta feature name as documented.

​
Error handling
If you use an invalid or unavailable beta header, you’ll receive an error response:


Copy
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Unsupported beta header: invalid-beta-name"
  }
}
​
Getting help
For questions about beta features:

Check the documentation for the specific feature