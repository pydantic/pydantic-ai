# Quick Start
This page contains instructions on how to set up your account and run your app with AI Gateway credentials.

## Create an account
Using your  GitHub or Google account, sign in at https://gateway.pydantic.dev.
Choose a name for your organization (or accept the default). You will automatically be assigned the Admin role.

A default project will be created for you. You can choose to use it, or create a new one on the [Projects](https://gateway.pydantic.dev/admin/projects) page.

## Add **Providers** by bringing your own API keys (BYOK)
Pydantic AI Gateway allows you to bring your API keys from your favourite provider(s).

On the [Providers](https://gateway.pydantic.dev/admin/providers) page, fill in the form to add a provider. Paste your API key into the form under Credentials, and make sure to **select the Project that will be associated to this provider**. It is possible to add multiple keys from the same provider.

## Grant access to your team
On the [Users](https://gateway.pydantic.dev/admin/users) page, create an invitation and share the URL with your team to allow them to access the project.

## Create gateway project keys
On the Keys page, Admins can create project keys which are not affected by spending limits. Users can only create personal keys, that will inherit spending caps from both User and Project levels, whichever is more restrictive.

[comment]: todo add screenshots or video
