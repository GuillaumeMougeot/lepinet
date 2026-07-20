# Is there a way to build a fast, small app quickly for lepinet inference

Goal: prediction app using the camera or the photo galery of a phone or desktop
and giving a species, genus and family names of a lepidoptera photo. 


Additional goals: The app also gives model confidence and highlight the levels
above a minimum confidence threshold, the other levels are greyed. If no family
is above threshold, then the app question the presence of a lepidopter or state
that the displayed species may not be part of the training dataset. For the
lowel level above threshold, the app display three pictures within its training
database. For each level, the app also gives the hyperlink toward the GBIF page.

The app should be light and fast and run offline.

## User interface

The app could look like a Google translate app. 

When launched, you get a home page with text explaining what this app is and in
the middle, a big camera button that opens the camera of the user and a small
galery button on its left side that open the phone galery. This could also be
just a camera button and when the camera opens, there is a galery botton on the
side. Ideally, rely on the default camera app, so the app does not have to
include camera management such as rotation, zoom etc. At the bottom of this home
page a link redirect a user who would like to learn more or give feedback to the
GitHub page.

After a picture has been taken, it shrinks down at the top tier of the screen
and the prediction appears almost instantenously at the bottom of the screen. A
loading symbol could be necessery in the future while waiting for the
prediction, but it would be great to not need it. 

In the middle tier, the predictions family, genus and species names are
presented on top of each others. There names can be copied to be searched. The
user can either highlight the text or a little copy symbol can be used. A
confidence score is displayed next to each level with visual cues. For instance,
this confidence number between 00 and 99 could be included inside a progress bar
circling around it that goes from 00 to 99 and from red to green. The taxa name
is greyed if the level of confidence is below a certain threshold. This
threshold could be 0.5, but this may change in the future. Next to each taxa
name, a hyperlink symbol redirect toward GBIF page associated with this taxa.
This 3x3 informations should be cleaning displayed in a structured fashion.

In the bottom tier of the display, three sample pictures of the lowest
highlighted level are displayed. They can be clicked to be displayed bigger. A
text below inform about the taxa name.

The user can then use the return button of their phone to go back to the home
page.

## Roadmap

This section describes the roadmap of the development process of this app.

The app should be as fast and as small as possible and run locally on the user
phone or desktop. Fast has higher priority than small. But both are important.

### Shrinking the model down to the fastest and smallest possible size

This section tackles the difficult problem of making a trained deep learning
model production-ready: fast and light-weight. 

The current model is a fastai-trained efficientnetv2_s with ~12000 classes. The
model heads is quite large. The model file size is currently ~170 Mb. The goal
would be to reduce it by x10 or more.

Repo used to train the model: https://github.com/GuillaumeMougeot/lepinet and
stored locally in ~/codes/lepinet. The script used is dev/030.

Here are some potential techniques to explore: To reduce the size of the last
layer:
- Low-rank factorization
- PCA on the logits -> Potential risk of strong accuracy loss.

To reduce the size of the entire model:
- Knowledge distillation
- Quantization 
- SVD?

Techniques with less potential or more substantial work/exploration yet: 
- Trained other architecture?
- Exploit the hierarchical structure even more by having a series of models?
- Learned embeddings: this is probably not going to reduce the size of the final
  app file.

More?

The part of the project is allowed to have time and space, meaning that the goal
has to be reached. Training new models is allowed. New scripts can be created
for this purpose in the dev folder, following the numbering structure. This is
to be seen as an engineering challenge and the app can go down in size as much
as possible. Less than a Mb is the dream. Go down to assembler if required
(well, this could also obfusticate the code so this could also be parallel dev).

### Creating the app

The app is intentionally simple and light-weight to be able to be easily
downloaded and run locally (offline).

The goal is to use most of the modern tools and techniques to reach this goal.

To avoid having to set the app on a "store" and be platform-agnostic, the
current avenue is to use PWA. This is strongly open to improvement if needed.

The app would be hosted on GitHub Pages, created with CI with GitHub Actions.
When opening the GitHub page, the user will be prompted to "download" the app,
like "'Add to Home Screen' to install this app for offline use." 

I don't know if this app code belongs to the same lepinet repo or to another
"lepinet-app" repo?

Here is an example of list of potential tools to use: 

| Purpose    | Recommendation                    | Why                                              |
| ---------- | --------------------------------- | ------------------------------------------------ |
| ML         | PyTorch                           | Already there                                    |
| Export     | ONNX                              | Browser standard                                 |
| Runtime    | ONNX Runtime Web                  | Fastest mature solution                          |
| Language   | TypeScript                        | Modern standard                                  |
| Runtime    | Bun                               | Yes                                              |
| Bundler    | Bun                               | Bun increasingly replaces Vite for many projects |
| UI         | Svelte 5                          | Small runtime, simple                            |
| PWA        | Bun                               | Offline install                                  |
| Hosting    | GitHub Pages                      | Free                                             |
| CI         | GitHub Actions                    | Automatic deployment                             |
| Formatting | Biome?                            | Replaces Prettier + ESLint                       |

This list is strongly amenable and must be changed if a row is wrong, too much
or missing. I can think of tools such as WebGPU, WASM or Rust lang.

### Future avenues of development

- Storing capture images locally or having some sort of identification history.

## Development rules

- Journal all developments: anyone reopening the code must be able to understand
  both the final product and the journey that led to it with all the important
  lesson. Keep an exhaustive but efficient writing style. If done within lepinet
  repo, follows the journaling method: create a journal entry per problem and
  document each new script in the dev/readme. 
- For dev, use all modern methods, uv, bun etc.