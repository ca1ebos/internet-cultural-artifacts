# Publishing Internet Cultural Artifacts

The GitHub project contains code, aggregate study results, and the CARLB benchmark and static viewer. The Hugging Face staging area contains the large model files separately.

## 1. Push the GitHub repository

1. Create an empty GitHub repository named `internet-cultural-artifacts`. Leave GitHub's README, license, and gitignore checkboxes unchecked because this project supplies its own files and history. Choose a public repository if you want a public Pages site on GitHub Free. [GitHub import instructions](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows)
2. In PowerShell, open the existing project folder. Run the following after committing any local changes you intend to publish.

```powershell
$githubOwner = Read-Host 'GitHub username or organization'
git status --short
git remote -v
git remote add origin "https://github.com/$githubOwner/internet-cultural-artifacts.git"
git push origin HEAD:main
```

If `origin` is already configured, verify its URL before pushing and skip `git remote add origin`. Complete GitHub's sign-in prompt if Git requests authentication. The push publishes the current commit as `main`.

If you received the source ZIP instead of the existing repository, extract it into a new folder and run `git init -b main`, `git add .`, and `git commit -m "Publish Internet Cultural Artifacts"` before adding the remote.

## 2. Publish the dashboard with GitHub Pages

On the GitHub repository, open **Settings → Pages**. Select **Deploy from a branch**, then choose **main** and **/docs** and save. Open the URL shown when deployment finishes. The dashboard files use relative asset paths, so they work under the repository path. The site has its own static data and requires no model service or token. [GitHub Pages publishing instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

Open a reply such as CARLB-037 and check that headings, lists, and quotes render. Then check a highlight link that jumps to the matching comparison.

## 3. Upload the models to Hugging Face

The staged folders are under `artifacts/releases/huggingface/`. Each contains an IQ4_XS GGUF, a model card, the upstream Qwen license and provenance notice, and a checksum manifest. The five fine-tunes also contain their PEFT weights and portable adapter configuration. The optional base-control upload is a quantization of the pinned upstream Apache 2.0 model. [Upstream Qwen release](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554)

The five fine-tune cards leave the fine-tune license unassigned. The study's source records document a private-research basis and source-specific terms. Before making a fine-tune public, confirm that you can redistribute its weights and choose the license that applies to the fine-tune. The bundled license file covers the upstream Qwen material. [Source records](sources.json)

Use the existing study environment's Hugging Face CLI. Get a write-capable token in your account's token settings and enter it only in the sign-in prompt. [Hugging Face authentication](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) · [Token settings](https://huggingface.co/settings/tokens)

```powershell
$hf = (Resolve-Path .\.venv\Scripts\hf.exe).Path
& $hf auth login
& $hf auth whoami
$hfOwner = Read-Host 'Hugging Face username or organization'
$model = 'qwen4b-base'
$repo = "$hfOwner/$model"
& $hf repo create $repo --private
if ($LASTEXITCODE -ne 0) { throw 'Repository creation failed. Check its visibility before uploading.' }
& $hf upload $repo ".\artifacts\releases\huggingface\$model" . --repo-type model
if ($LASTEXITCODE -ne 0) { throw 'Upload failed. Check the CLI output before continuing.' }
```

On a current standalone CLI, replace `hf repo create` with `hf repos create`. The folder upload handles the large model file. If the repository already exists, verify its visibility in **Settings**, then rerun the upload command directly. Opening the model repository also lets you verify that the card, GGUF, and checksum manifest are present. [Hugging Face CLI upload](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-upload)

Repeat the create and upload commands one model at a time after setting `$model` to the desired folder name and recalculating `$repo`.

| Model | Folder and repository name |
|---|---|
| Optional base control | `qwen4b-base` |
| Stack Exchange | `qwen4b-closedasduplicate` |
| Yahoo Answers | `qwen4b-yahooanswers` |
| Quora | `qwen4b-assomeonewith30yearsofexperience` |
| LinkedIn | `qwen4b-humbledandhonored` |
| Reddit | `qwen4b-majorredflag` |

The six folders together contain about 14.05 GB of weights. Check available private storage in the destination account before starting. [Hugging Face storage limits](https://huggingface.co/docs/hub/en/storage-limits)

The cards use `<HF_OWNER>` in the portable Python loading example. Substitute your account or organization there when you prepare the final card. Add the GitHub repository and Pages links once their URLs exist.

### Restage from local study artifacts

Run `python .\prepare_hf_release.py`. The script verifies the GGUF and PEFT files against the saved run digests and uses same-volume hard links for the large files. Keep the staging folder on the same volume as the original files. Editing or truncating the hard-linked weights would change the original, so edit text metadata only. Use a new output folder when restaging edited cards.
