# Google Earth Engine Authentication Guide

We've encountered challenges with the interactive OAuth flow in the cloud environment. Here are alternative approaches:

## Option 1: Service Account Authentication (Recommended for Production)

If you have a Google Cloud project with Earth Engine enabled:

1. **Create a Service Account:**
   - Go to: https://console.cloud.google.com/iam-admin/serviceaccounts
   - Create a service account
   - Download the JSON key file

2. **Use the service account in the script:**
   ```python
   import ee

   # Initialize with service account
   credentials = ee.ServiceAccountCredentials(
       email='your-service-account@project.iam.gserviceaccount.com',
       key_file='path/to/service-account-key.json'
   )
   ee.Initialize(credentials)
   ```

3. **Upload the key file to this environment** and update the script.

## Option 2: Local Authentication + Credential Transfer

Authenticate on your local machine and transfer the credentials:

1. **On your local machine, install Earth Engine:**
   ```bash
   pip install earthengine-api
   ```

2. **Authenticate locally:**
   ```bash
   earthengine authenticate
   ```

   This will open a browser, you'll sign in, and credentials will be saved to:
   - **Windows:** `C:\Users\YourName\.config\earthengine\credentials`
   - **Mac/Linux:** `~/.config/earthengine/credentials`

3. **Copy the credentials file** from your local machine

4. **Create the credentials in this environment:**
   ```bash
   mkdir -p ~/.config/earthengine
   # Then paste the credentials content
   ```

## Option 3: Direct Browser-Based Authentication (What We've Been Trying)

The challenge with this approach in cloud environments is that the OAuth flow requires:
- Interactive terminal input
- Proper code_verifier/code_challenge pairing
- Correct OAuth endpoint parameters

The errors you're seeing ("invalid request") suggest the OAuth parameters aren't being accepted by the Earth Engine authentication server.

## Recommendation

For testing purposes, I suggest **Option 2**:

1. Run `earthengine authenticate` on your local machine
2. Copy the credentials file content
3. Tell me the content (I'll help create it here)
4. Then we can run the gold detection analysis

OR

If you prefer not to share credentials, I can:
- Provide you with all the scripts to run locally on your own machine
- You can run the analysis locally where authentication is easier

## What Would You Like to Do?

A) Try the local authentication + transfer approach
B) Set up a service account (if you have GCP access)
C) Run everything locally on your machine instead
D) Continue trying to debug the OAuth flow

Let me know which approach you'd prefer!
