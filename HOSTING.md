# Putting the Strain Lab online (free): Streamlit Community Cloud + your private Hugging Face dataset

The app runs on Streamlit Community Cloud (free). Your data stays in the private Hugging Face dataset
`Ryan-Johns/strain-lab-data` (free). Everyone who signs in with the password works on the same data.

## 1. New Hugging Face key
huggingface.co > Settings > Access Tokens: delete the old token, then **Create new token** > type **Write** > name it
`strain-lab` > Create. Keep the page open; you'll copy it in step 4. Don't paste it anywhere else.

## 2. Put the code on GitHub
1. Make a free account at github.com.
2. **+** (top right) > **New repository**. Name: `strain-lab`. Choose **Public** (the code has no data or keys in it)
   or Private. Tick **Add a README file**. **Create repository**.
3. **Add file > Upload files**. Unzip `strain_lab_github_upload.zip` and drag **everything inside it** onto the page
   (`app.py`, `requirements.txt`, `README.md`, `.gitignore`, and the `strain_lab` and `.streamlit` folders).
   **Commit changes**.
4. Check that the `.streamlit` folder appears in the file list. If it doesn't: **Add file > Create new file**, name it
   `.streamlit/config.toml`, paste the contents of that file, **Commit**.

## 3. Create the app
1. Go to share.streamlit.io and **Continue with GitHub**.
2. **Create app** > **Deploy a public app from GitHub**.
3. Repository: `<your-github-name>/strain-lab`, Branch: `main`, Main file path: `app.py`.
   App URL: pick a name, e.g. `strain-lab-uconn`.
4. **Advanced settings**: Python version **3.12**. In **Secrets**, paste (with your own token and password):

```toml
HF_TOKEN = "hf_...your new token..."
STRAIN_LAB_DATASET = "Ryan-Johns/strain-lab-data"
STRAIN_LAB_PASSWORD = "choose-a-password"
```

5. **Save** > **Deploy**. The first start takes a few minutes. It then shows the sign-in page.

## 4. Share
Send your advisor the app link (`https://<name>.streamlit.app`) and the password.

## Good to know
- **It sleeps** after a while with no visitors. The next visitor sees "This app has gone to sleep" and clicks
  **Yes, get this app back up**; it wakes within a minute. Nothing is lost.
- **Saving**: changes are pushed to the dataset a few seconds after you make them. The sidebar says
  "All changes saved online".
- **Memory is about 2.7 GB**: upload one location's ZIP at a time, not several GB in one go. If the app runs out
  of memory it restarts; anything already saved online is safe.
- **Updating the code later**: in the GitHub repository, **Add file > Upload files**, drag in the new files
  (they replace the old ones) and commit. The app restarts with the new version by itself.
- Changing the password or token: share.streamlit.io > your app's **⋮** > **Settings > Secrets**.
- Use the online app for shared work. Don't point your PC copy at the same dataset at the same time.
