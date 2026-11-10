# Model license boundary

Reviewed November 10, 2026 against the official [MiniMax-H3-Image license at revision 492fd8c4eca666f1085f57e30738c1e0d86b3439](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/492fd8c4eca666f1085f57e30738c1e0d86b3439/LICENSE). The packaged source lock records its SHA-256.

This is a practical source review, not a legal opinion or a finding that a particular user has permission.

- I.5 identifies the European Union, United Kingdom, Republic of Korea and United States as excluded territories. II grants rights only within the applicable territory. V.4 also restricts use, reproduction, modification, distribution and display of the works and their outputs/results outside that territory. II describes contacting MiniMax for separate licensing in excluded territories.
- IV.1 requires **separate prior written authorization** if the user's/affiliates' annualized revenue exceeds USD 1 million **or** a hosted service is provided to third parties. Hosted service is broadly defined in I.8, including API/web/remote access. This differs from the old H3 license and is not limited to paid hosting.
- The materials and model derivatives are defined broadly in I.6/I.10/I.11; adapters, model-derived training and related inference material are not automatically exempt. III has redistribution and notice requirements. V includes acceptable-use, downstream terms and safeguards requirements.
- The Comfy-Org card explicitly links to this new image license. Repacking or quantizing the weights does not establish a different permission grant.

This repository contains newly written planning and software-check code, not model weights or a copied native MiniMax implementation. Its MIT source license does not relicense any model material. Reading public metadata did not establish the user's territory, revenue, separate license, or permission to host. No acceptance or grant is claimed.

Native operations are unimplemented and unavailable in this release independently of authorization. A future native integration needs both applicable rights and engineering qualification. An `authorized=true` flag or an acknowledgment alone would not implement missing functionality or resolve legal applicability.

No model download, training, rendering, model redistribution or deployment occurred during this setup. No automatic license-acceptance flow or hosted endpoint is included.
