# What the blocked servers cost, what substitution recovers, and who to ask

**Kind:** research · **Status:** **RESOLVED (2026-10-01), pending the dead-host pass.** Of the
87,560,065 images selected from TreeOfLife-200M, **20,105,238 (23 %) sit on nine servers that refuse
automated access**. Re-pointing the 2,000-per-species cap at accessible servers recovers **4,534,176**
of them. What remains lost is mostly **herbarium plant sheets**: for Lepidoptera, **89 % of images
survive and only 38 of 19,150 species are lost entirely**. Contacting two citizen-science
platforms -- observation.org and Artsdatabanken -- is worth doing; chasing the herbaria is not, for
this project. Along the way: a content-type bug that discarded valid images, three dead servers, a
folder-naming bug, and the discovery that 9.3 % of the "species" keys are not species.

## The question

The owner's idea: the cap takes 2,000 images per species, so if some of those sit on a blocked
server, other images of the same species -- never selected because of the cap -- may sit on servers
we can reach. And if the loss is small or irrelevant, accept it.

## The cap, confirmed and qualified

`plan --min-img 50 --cap 2000`: species with at least 50 images in the whole catalog, at most 2,000
each. The qualification that makes substitution possible: the cap takes the **first 2,000 in
catalog order**, blind to the server -- and the catalog is ordered by server. A common species
fills its 2,000 from early servers and its blocked-server images are never selected; the selected
blocked images belong disproportionately to species with **few images elsewhere**, which is why
substitution recovers less than the raw surplus suggests.

## Method

`substitute` replays the plan's selection row by row over all 233 M catalog rows. Its running counts
matched the original plan log digit for digit at every 100th row-group, and its total is exactly
**87,560,065** -- the selection is reproduced, not approximated. For each selected row on a blocked
server it emits the next unselected row of the same species on an accessible one. Run on a
workstation: zero UCloud core-hours.

## Result

| | images | |
|---|---:|---|
| selected | 87,560,065 | |
| on blocked servers | **20,105,238** | 23 % |
| substitutes found | **4,534,176** | recovers 22.6 % of the blocked |
| still lost | **15,571,062** | 17.8 % of the selection |

| | species |
|---|---:|
| lost entirely (no accessible image) | **1,410** (0.7 %) |
| pushed below the 50-image floor | 45,243 |
| kept at >= 50 accessible images | 157,225 |

Substitutes land mainly on the iNaturalist bucket (2.88 M), Artportalen (276 k) and Flickr (144 k):
fast, accessible, and field photographs.

## What is lost is mostly not what we need

| blocked server | images | what it holds | Lepidoptera |
|---|---:|---|---:|
| `mediaphoto.mnhn.fr` | 5,246,856 | MNHN Paris herbarium (plants) | 0 % |
| `observation.org` | 4,491,443 | European citizen-science field photos | **17 %** |
| `content.eol.org` | 3,365,698 | EOL, mixed | 10 % |
| `medialib.naturalis.nl` | 3,273,698 | Naturalis herbarium (plants) | 1 % |
| `oxalis.br.fgov.be` | 1,510,405 | Meise herbarium (plants) | 0 % |
| `images.ala.org.au` | 1,227,932 | Australian citizen science | 7 % |
| `www.artsobservasjoner.no` | 774,956 | Norwegian field observations | **17 %** |
| `www.antweb.org` | 114,981 | ants, specimen photos | 0 % |
| `scan-bugs.org` | 99,269 | pinned insects -- **server retired**, not blocking | 66 % |

(Lepidoptera share from a 50 k-row sample of each server.)

By kingdom the blocked images are **63 % Plantae**; by basis of record, **10.7 M preserved
specimens** against 1.4 M explicit human observations (observation.org's and EOL's records carry no
basis in the catalog; observation.org is in substance field photography). The three herbaria alone
are **10.0 M of the 20.1 M**.

For **Lepidoptera**: 8,779,051 images planned over 19,150 species; 1,571,149 blocked; 601,185
substituted; **~970 k lost (11 %)**; **38 species lost entirely**. Lost species overall: 870 animals
(392 insects, 190 arachnids), 389 plants.

## Recommendation

**Accept the herbarium, EOL, ALA and AntWeb losses for this project.** They are mostly specimen
material in a domain far from field photographs of moths, and for a general backbone they are a
minority of a large corpus. The one caveat is D2's matched-corpus comparison against BioCLIP-2,
which was trained on everything: our crawl is a strict subset, and the paper should say so.

**Contact observation.org and Artsdatabanken.** Together ~5.3 M images of European and Scandinavian
field photography with ~17 % Lepidoptera -- the closest thing in the whole corpus to the Danish
trap deployment. They are also the two most likely to say yes: both are citizen-science platforms
built on open data.

**Optionally ask Imageomics.** TreeOfLife-200M was downloaded with their MPI
`distributed-downloader` on an HPC cluster (up to 20 nodes x 20 workers, default 3 requests/s per
server, an ignored-server list); neither the dataset card nor the paper mentions agreements with
providers, and the card tells users to re-download the images themselves. They evidently obtained
observation.org's images when they crawled, so they can say how -- and they hold 224 px shards, though
licences probably prevent redistribution.

### Contacts (GBIF-registered unless noted)

| server | operator | contact |
|---|---|---|
| observation.org | Observation.org (NL) | info@observation.org |
| www.artsobservasjoner.no | Norwegian Biodiversity Information Centre / Artsdatabanken | postmottak@artsdatabanken.no |
| medialib.naturalis.nl | Naturalis Biodiversity Center | informatiemanagement@naturalis.nl, collectie@naturalis.nl |
| mediaphoto.mnhn.fr | Muséum national d'Histoire naturelle | gildas.illien@mnhn.fr |
| oxalis.br.fgov.be | Meise Botanic Garden | quentin.groom@plantentuinmeise.be |
| content.eol.org | Encyclopedia of Life, Smithsonian NMNH | secretariat@eol.org (EOL forum) |
| images.ala.org.au | Atlas of Living Australia | support@ala.org.au |
| www.antweb.org | California Academy of Sciences | jfong@calacademy.org |
| scan-bugs.org | SCAN -- retired | none; no mirror at scan-all-bugs.org (404) |
| (dataset) | Imageomics / TreeOfLife-200M | dataset Discussions tab on HuggingFace; curator Matthew J. Thompson, thompson.4509@osu.edu |

## Found on the way

**A bug that discarded valid images.** The crawler rejected any content type outside `image/*`, but
CDNs serve JPEGs as `application/octet-stream`: **52 % of `d2seqvvyy3b8p2.cloudfront.net`'s images**
were being thrown away, and the same check ran through the whole first crawl. Fixed (only `text/*` is
rejected on its header; bytes decide), and `--retry-status bad_content_type` recovers what earlier
parts recorded as failed -- verified on a part where it restored 2 of 2 and kept all 40 records.

**Three servers are dead, not blocking:** `sernecportal.org` (HTTP 404 for every image),
`files.plutof.ut.ee` (the name no longer resolves), `storage.idigbio.org` (expired TLS certificate).
Previously every row on them was tried with retries; a dead-host breaker now stops them after 200
requests and records the reason, and they were found by the crawler itself within minutes of the
restart. ~572 k selected images; a second substitution pass covers them.

**Not every "species" is a species.** The key is `genus + ' ' + epithet` and degrades when a field is
empty in the source:

| key kind | keys | images |
|---|---:|---:|
| binomial species | 184,853 | 80.4 M (91.3 %) |
| genus or higher only (`Megaselia`, `Sciaridae`) | 12,915 | 4.8 M (5.4 %) |
| epithet without a genus (`occidentalis`) | 6,110 | 2.9 M (3.3 %) |

The last row is exactly the failure the plan's own comment warns about -- a bare epithet merges
unrelated genera; `occidentalis` alone is 129 k images -- and it happens whenever the genus is empty.
The honest species count is **184,853**, not 203,878. Genus-only images are usable at genus level in
a hierarchical model, which is lepinet's design; bare-epithet ones must be excluded or re-labelled
from the catalog before training. Downloading continues for all three: it costs almost nothing on
1 vCPU and labels can be fixed later by joining on `uuid`.

**Species folders are mangled.** `slug()` has no `.lower()`, so every genus initial became `_`:
`Acacia dealbata` lives in `_cacia_dealbata/`, and 158 folders are shared by two names. No image is
lost -- filenames are unique uuids and the meta records each image's species -- but the folder name
must never be used as a label. Left as-is mid-crawl for consistency, and documented in the dataset
README.

## The dataset README

`/12383016/treeoflife_200m/README.md`, generated by `dev/082_tol_crawler.py describe` from the
manifest's own JSON so it can be refreshed as the crawl proceeds: layout, schemas, the selection
policy, image processing, composition by kingdom/class/basis, the label caveats above, the blocked
and dead servers with contacts, what they cost, licences, and how to reproduce.
