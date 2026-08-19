# Cloudflare deployment

Wav2mid HQ is built for Cloudflare Workers Static Assets. Transcription still runs in the browser; Cloudflare serves the application and lightweight runtime/model assets.

## Permanent production deployment

Configure these GitHub repository secrets:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

Then push to `main` or manually run **Deploy Cloudflare Worker**. The workflow builds, checks the permanent 25 MiB/file asset limit, runs `wrangler deploy`, and writes the deployed URL to the job summary when Wrangler prints one.

Local equivalent:

```bash
npm install
npm run deploy
```

## Temporary claim deployment

For an unauthenticated preview, use Cloudflare's temporary/claim deployment flow:

```bash
npm install
npm run deploy:temporary
```

The script performs the normal production build, then applies an additional **5 MiB per-asset gate** before running:

```bash
wrangler deploy --temporary
```

Wrangler prints a preview URL and a **claim URL**. The claim URL is a bearer credential. Do not paste it into a public GitHub Actions log, issue, PR, or chat with unintended recipients. Cloudflare requires the intended user to claim the temporary account within the stated deadline if the deployment should be kept.

`deploy:temporary` is intentionally a local/manual command rather than a public GitHub Actions job because the claim URL must remain private.

## Neural HQ assets

The optional HTDemucs mode does not place the large neural separator model or ONNX Runtime WebGPU WASM in the Cloudflare static bundle. Those are fetched only when NEURAL HQ analysis starts. This keeps both permanent and temporary Cloudflare asset limits satisfied while ordinary FAST/PRO/INSANE use remains lightweight.
