# Provenance

This file lists the sha256 of every file in `audit/`, states what differs from the copy in the
experiment tree where the work was done, and names every artifact whose generating script is not
recorded inside it. `SHA256SUMS` at the repository root carries the same digests in the format
read by `sha256sum -c`.

## Stamped artifacts: notes translated, payload unchanged

The JSON artifacts had their free-text notes translated into English, with references to the
project's internal review steps removed and internal-note file names replaced by neutral ones
(`internal/note_NN.md`, `internal/design_note_01.md`, `internal/library_record_NN.md`). No number,
boolean, key, structure, dotted field pointer, fact id, artifact name, short label or sha256 changed.
`verify_payload.py` checks this: the payload digest of each published artifact (canonical JSON with
free-text strings masked) must equal the digest of the original, listed in `PAYLOAD_SHA256` and in
the last column below. A sha256 recorded inside a stamp (`_fontes`) refers to the ORIGINAL bytes of
the file it names; the first column below gives those, so the chain can be followed across the
translation. The two Parquet tables are byte-identical to the originals.

| file | sha256 of the original | sha256 published here | payload sha256 |
|---|---|---|---|
| `_arquivo_2026-08-09/auditorias_pontuais/qualidade_lidar_4biomas.json` | `cb46d7a7e8350b210c1dd05c8602f9e455761e643303b41af3c3b0998bde8e50` | `6c488b75d9aa4c3081dad0f0e70c14d3489c4dee24077104a9e3fba6303a9a4e` | `fc3d270e223d8b5c7cf035a05d52a1d47db8761d613f18f9e8cb4dfb956a7c88` |
| `_arquivo_2026-08-09/auditorias_pontuais/qualidade_lidar_do_zero.json` | `b42a7050b0d6eb5b5bcbf43db4a0f2b00af090d1c5d5b9b04785e0af7a79c261` | `30837dfce62ba1dbe91ddeb03c464f13024ea6eec99b1284bbe99c690ae08334` | `bbe32f83a7276ac74d8b7af0e7babcbd3c66c0bcc668945c629daae6b63c1efd` |
| `_pareceres_2026-09-28_execucao/geo_produto_anadem.json` | `2738c9edce6e55fa5076e08c4479bfe262c7a6788b2b5730ec1f5edd63c913e6` | `7caa26f9614f0d6ac8169b924704f7721d38dac7e32c1ca2310058767ff65534` | `20b126e643696f80aae2424783772b45513e057f1635dd9aac870ff8a6e97749` |
| `_pareceres_2026-09-28_execucao/guarda_sanidade_anadem_vs_glo30.json` | `fd35875b32154733ab36c9f0b4aa5b827d8f7f8cb86c0833c1e85827a39e73ee` | `c6fc28923fbc5e27a15793c9cbf15729f5345ae96f2811ed616a8eb3f6a26d65` | `1d3c934e8f61fc34b9500187aa1bcb985654556b787737992e46e41cdab941bc` |
| `auditar_admissibilidade.json` | `b7ece3993e06263502771234d707a32c911370858bf6ce02e66f6e5082273b86` | `cb547f8d3b10352093cc386188bedde454fb0c5c6732214b4e69fb0f83db20e9` | `c04fe3a24a4d96de1839674f5446b793865f0efeada5b7d85a2f81228723eb51` |
| `auditar_d5_invariancia.json` | `1b844ec743c78314e31d28105899c4c405f24b9faaf860b7fcfc9d54bda2d369` | `95afc82e0c1b580a1f09e6685d7d61b063b4cb33c169eb1bbd23b6ff567bef01` | `f96258153c7b7d152eb2f949056fb68eeba91625fbea7fc0d67c320b5c6abd17` |
| `auditar_d8_triangulo.json` | `8c034686e697d6cb82bdf233e738c3e061cf3726367a9fbfbbc922766e607bc5` | `8d6c0ffa8244e4f895c2fb41174e2d0ccf260efd5e867be7c3130b7da375875c` | `1ead871f1cf5e391a5e73d9f6e68443f43179e4b0fce9d993432bde1354c3c25` |
| `auditar_densidade_solo.json` | `4f57ca86d3492bdddc39442ffe9abc99f9caad8595317835e296d817aee2ae00` | `73931d111cbdfaa41d46043afa5db3ffefb70f36587ce4e992cc4c2c4e3aab10` | `00b34c203510965a193700bb24aab4d28c9e98e03346474f4754f15731e55d53` |
| `auditar_densidade_solo_nativa.json` | `e9eeb3634ec1fde6307ed11048e6063e019101f2ae96e9942d4c471de96bbe3e` | `c34ca1707a72b4532513e3be888e9385da1e14eba4a3d230ed68d63e6893ea25` | `4b91b67a5979058fde3a117e82b8a2129129d2dd144ff41d5f115c1c0ce191ca` |
| `auditar_dt3_criterios_registro.json` | `ecb7f3449e7506b5f4faf84fae13ba85cfd6f5f404bbeaba00f1b1f01b37ddb0` | `4b88928c735f2194dbffbf14e0e206f94cbf0ccd0e801b60ab73ba7cdf04e4c3` | `4891865ddc828f204c64a19e1e278264a0e018c79245061cfd5c31e8f660efc6` |
| `auditar_f1_rampa.json` | `a2d47c6ec1265a831d28029c8b508043f775cc43cbbcf9beb2e3af6a21db2d24` | `c5d66c74adec8bf32d3df78585957355f8d21bbe29c5a23f212711239183087b` | `189abc320e1ca7c63be15d0295c774330a08c45219b481cd2f6d74ddab277196` |
| `auditar_f2_celula.json` | `16a603cdbcc1c8d6c4e9aa9dc0b92ee085e77997b52a9fafe89f5ea6218b3a01` | `358ce1ce3b96260eb8488582d4ba8c959b99b0f85a508d66f5236631d6550353` | `533c5e7d0a418b6b72750059d8676ae4f539a5a693ba8cec27bd9ef65d82812f` |
| `auditar_lote1536.json` | `82e5fa792fab97aa1e9147245dd5e14506da9c168f7ac9b5fa47490946faf259` | `123c4c9d16430d3f93a9c46ccd084c303419d36c5ececceada43e55cc42dc249` | `d6c5dcff5288072d45810332fb6f2900a6fd65c883a4f536fcb677e5c37a2f7b` |
| `auditar_nmad_pareado_nativa_anadem.json` | `95c726a1a7b703ba0092b74560df34e1b2ebb8225a45567ce7ffbd641aa75b66` | `88041dd73ec74d6ff0b1ee2fbe147e3f4625cff71f44aae111d710fbd3d20a06` | `62c3a0e422760ff9983695fc73ac093f18ea1a19829ddbc1814a34ce6e944a94` |
| `auditar_nmad_pareado_nativa_diretas.json` | `bb3cce3e12470fb2da78fb3ebcc25fb3beeffee7a10c637726eaf3f7e6f00f2d` | `d89d6696c91e0ddc9ac01af847b7255f94ec3681f9e20765d1a0e7cec9cebb4d` | `4c6a24815423c1602d56bdab295780ab6e2f7edbb00c77212ecfbbbc59c67ce5` |
| `auditar_paridade_capacidade.json` | `448bedb643f8dd1355bd06d3fd123a78c904111e6c30ffe6410e2593c207d3a2` | `352216e948a7292159d9dd3573ef34500e88ba46a4101a977896b9bcc89a1b98` | `165675e0786eea6f9c6c6f24f4d129dfa717f1547a14848c8d7936dcfc833081` |
| `auditar_perfil_anadem_faixa.json` | `4325c5e3b63a339b25293b745580650fc458a41c28ea545e67816d16d069f405` | `c6ace9abafaf0292485409097430312c6fb4208ee62a4b2cfa728ec8b21b6628` | `32ce51d4acdb6b8c0d25c7e86c4657dc20f3fc036671423963735b68684b27cc` |
| `auditar_recursos_s7.json` | `93a96efff7bcb49cf1eb05583dfff5f51fd02bbab6dcecb512161f2e8e44e986` | `01950374b53168488115ac8c33ccc388531135dab2354636fd787af50f68cc98` | `78255af2f5bba6da59f7c795bb213fe4e3673a27f403f2957eb80a3e36e7a666` |
| `auditar_rf_filtro_gedi.json` | `1334a3e80966e416bbe6822248a731afd065e84c381db425f9b0c5ac61827306` | `90deb090bfe2d4c0276420d718960a0a336936f75822523b6933a2987d5f58be` | `57d0c2671b8152aadbc67cca13c5ca5079d5647c0c3f5e88be4efac82b36e940` |
| `auditar_rf_filtro_gedi_v2.json` | `48b1698c371d9ed8c1700f842de9cf37b5e63ac08b092636cc1f5643bceaca91` | `74ab6e13e200acd8a8402352cbc12fddf06485423368f09a7fa1098cd820b727` | `5b98da3cafce028e2cb502e5020bdb62f571ff15e3443635137243205647b740` |
| `auditar_robustez_o3.json` | `a49916313c23aa54ccfb70a6461b346cf601c5adb69e971a16632e8196e8d7e1` | `f7e21da632b458c8ebda01b40536d719dd0caaa32096cd6133cec89dcbd9501e` | `7c439d59d4a508982471490385f6bbef1b33481b578b217ff68a2f48f252b477` |
| `auditar_rotulo_regua_v2.json` | `2470115aaf6eed5ff2ea7d878f2dd6a1ac3f7bf682240599102a33fb01129e9a` | `e02a5131bf9962e34f71b6c9feb9b292eb111919c4d7644f1de43ba8a413c31b` | `620cb79fae7e75212ddded0321a0818b727fd00e6676fa4c6accd16680866d88` |
| `auditar_s1_estratificacao_lidar_v2.json` | `30d529bd422f105e1f8ab8e9dacf74b6fdacdf3de241d3b93a1a8a6524598f77` | `1274849694c3bf9eb7edf009ea2eeca7437d3ed59c8cb9019946c5522396367e` | `0a2a598746126fee94662cf1fd6d54591cfcf988b42be90d52f43fa9c654671b` |
| `auditar_s2_registro_referencia.json` | `aa3f0f34b2965da9f879b060f2beccb39d994fb2c2ac7bbb8c5674e33742a7a2` | `d677a1953e5b2d084e4fc12ba79ceefbf08e35a184157bd586b0a6abbb6bfd9f` | `9be6db958cc1b0b9fc2df0ec2a084fe4104c0ccecea81f97fbef53c3198a7454` |
| `auditar_s5_baseline_fisico.json` | `8f62b6a346692783614dc806034a36b2aa1773b057c0f74cb818e7f6df9156cf` | `e49a341e4384080c8e7f95209c339a69d2378b3629c116c88212af9eb890db9d` | `9d32c88aef23f27514a810245ed660e089c7efed643a5e9191c11ac5ee411830` |
| `auditar_s7_arvore_lidar.json` | `09d3030b23bfe6d908cd9bc4b9832418b4872b36aa641198a6faf6007c9abf57` | `3ad496d727f0267948666c3b871811f1a92b2da9014c0d922db4aacd7d416c03` | `253c827b1b02e25e3c85629573f7f41b5f32be2033e407cecce389ae951ef0ba` |
| `auditar_s8_mascara_disturbio.json` | `943e4c47da8e77a7e8970aa7d55f1cccdfef4cceb92364f84f51d1db8febe92b` | `db99aade3264036b60d887f2a2d6782d063a2e2c4f2757861aeca3e2bd7c6c2b` | `7ad45ad0c167e70c5cb48349c2135308881acb4c67cdf0862687a46cab5a74f6` |
| `auditar_tabela3.json` | `613cb716c8b9cd30a1d60046bf8d0c7cf3ead3ab471de7131565b342856e49d1` | `b805329deebd8750f6245d8de92d1cd48495e16d35950f1b782ee362b91202c6` | `e8e53bc8bad9fd6d42b00b5de0c70aa99c7d9f7e74d6b8e6e082855c47d7e713` |
| `auditar_tabela3_dump.parquet` | `652d60a6db96fc2c1ecf700214ce270e19d4ab20cfaf21b1876ba338881d9023` | `652d60a6db96fc2c1ecf700214ce270e19d4ab20cfaf21b1876ba338881d9023` | `n/a (byte-identical)` |
| `auditar_unidade_quadrante.json` | `e4800fcdedb67bcfc25d03c94a61e1ecbcf32c01c2795372e33ce7e68d37f029` | `0bdaf0499265217735cfa0bb65d6de7cb239047c5597283313a4a8cc7acdf921` | `9d124a7ce6a6407807b1ff2972b781138aafdfe9f098443329e990146e6a9b40` |
| `comparar_ponto_s035.json` | `8b86c87ada3738d85ce5f369e5dfb74f5bfc0c21626d58a5f9722efb2432fba2` | `eeaf3201366ec49221808cfc89679b343a81e7f173e74cb992fbc5bfb528d92d` | `8af42a89e2bc254a9b8be4c0fec3c52bdec481d09391303e4f07425fa197be0d` |
| `comparar_suavidade_fina_ba4.json` | `dd860227c347f4b73ae227aef9aaf2348dd875ea7f3bfeb1450fa8e2996d7378` | `39623a669ed0f131a9042117ac0ad7e3507efe9e44cbff6a832cb00934b75eea` | `dfae2b3195830dfe987a953c9516d7aeec8a5b5f084f8e11a11f6e3691833291` |
| `delta_escala_celulas_nativa.parquet` | `110d75ca56626f268f32ece2acb2d0719ff7a19ced3e8f2b18fa3c122b2a14c2` | `110d75ca56626f268f32ece2acb2d0719ff7a19ced3e8f2b18fa3c122b2a14c2` | `n/a (byte-identical)` |
| `f2_arvores_config.json` | `142aba794f5c2f530c118ad416dac33de060821b7614615d4607e02af5e4d2b2` | `89ced3baedddc2ede88167e76323fba7bad4dd8144e530973071a31ec2812dc1` | `9432f281d6c4f8dd899a0a6019ce4bfeae0d30314247d85ca31f77baa112c416` |
| `fig/fig1_numeros.json` | `abe6e93e6c94a65d672af8b3edb797cb96c889d66662e2eb730635aa8720bb90` | `d9c733badd7389718fbc19c3d25f68e579e9de64b2debddcb138fd352803a932` | `130ed109517d6e2bbbe4b6a9f4067eb590f054e7b631c86029af0bfe6f4c81c7` |
| `fig/fig3_numeros.json` | `d61302f37bbbe3e970032154185a06764d4a1f531de25a90330fe4a654a25e32` | `52dfcaac41c67b067d6af5426b542883d8afdfc4d6001644c8b733708d979b07` | `8d94e2c4e4a0324b0a35377be0110f26230dd3e13c74325b7ef64d567b6423dd` |
| `fig/fig9_numeros.json` | `f57be0565fa93a14ecd1dfc5458929e20739fe9dc8ae4bc22cb417f0ccb04548` | `31af61976b92e85f4ad03cbe7c12daf9c75b5dce07b438685b9fba83cc476271` | `ff76de33061821fb8fd816f5efaa2b258452a358db4fd6556e365dc5c6a947a7` |
| `folha_de_fatos_acdsa.json` | `848ccc1ec5afbf67c3f36f7a063ca4ecc604e0217c7bf69afa311285d6935207` | `c00c3331f440d9f743dcd58f6598292ef700d40217a35b037870d27ac811e2d9` | `eee19ac26376a819916d741263805da4b77b6e1eb11a2d755b47d8e6ba6f229f` |
| `folha_de_fatos_jstars.json` | `b8a16a421486c9ac37b1b249e1020b02e9fd1ec9753679bfc2bd234714669103` | `f2ac9e22f52d6487b2a2a05ef304666efc88e264926f90c58c2f0cf9a229be2b` | `44b9290546ecd2b59c10ab0dfc9fb2e9361895944b07f78734137816d8fd272b` |
| `folha_de_fatos_jstars_v2.json` | `e32782c40b05d7b19cd7e7097097e06ea1fa1383fbb9581a73138ab4cbcfb80a` | `26fbb49fc33dd524264cf1037c724f4b77108c96ec7c7a29ba34bc4fc100c4fb` | `e6752392c4afdd5ef530cb11bb11798d9895e99c37420de4d7eeaf2f107c822e` |
| `folha_de_fatos_jstars_v21.json` | `6104492b226225623618f5e1148d7b3aeba4fa3ae72e96352009ffbd5d0013fb` | `d1b54a0dd63f8f590bae4eb5f8f09239d6a8b7419da61cdc51fef3f372550766` | `1b6c08d2829a1ee476fb5212233dd3e4ce3d2e5587f4e2414eb0762b32368239` |
| `folha_de_fatos_jstars_v22.json` | `101d470477b08e307186bb9b41c564b25295b52fb4263ff067cc2ee0c56c0d8d` | `70c399e161882783769610b02675007c70d9d0169f3d45a6090e9a5efccec8a0` | `03464c8eb14ebbf4e5dc59f1fce9a85368aeb9f636ee60abfc31ade8c6aaab52` |
| `folha_de_fatos_jstars_v23.json` | `1dbce07c1ec76ff8812bc1c28b737ae9886bae36a933f374fadd312ad8f9e768` | `934fce4bba573a71839b3ceeb4f5af5871c651e7b784ded7002682e24dd98152` | `bc6cccfca2267556bed4d3335682ba975a809677f60365d3f389a3bae738b299` |
| `folha_de_fatos_jstars_v24.json` | `0c3d4c725d2d7de15feb4924096a3e9088af6cfcef4a50dbc6d43a9c405bdc1f` | `2be04161e67508454985878118c5d82df85e9d4dca7ae520bc5b88708f8866d2` | `617bee80b1d712e2d9b2bd2e95f5055d1e77c30d2939bc9f778f2558a170bb84` |
| `folha_de_fatos_jstars_v25.json` | `aa3c5e9577dd0d9b476248cdc8e0e6bb5b78cf374f4e0d5bbe4a040a56c16b99` | `dc35396787e626483236ccbc2e82816134e294935772faeb6173e76869483fcf` | `e1e0a78dbb89d68ef177a59800755f7e68246263417b0c6b509fb823a44d2ab9` |
| `folha_de_fatos_jstars_v26.json` | `7f8701e91988198746172f4f97d900858621a843826be06880bd0c22f36b8842` | `5584b61e73a4e1a008e2ba4a670ddf1b9f8b13b4415253728c37f2e2cdd085cb` | `26e774f972af79c263fefb4037001d1da277e37520ae7a1741f9750e8e9a279f` |
| `folha_de_fatos_jstars_v27.json` | `dfa4770ee5bf801e14875a4cbaf7453e4dafb7593734f230c350f9a4c31b3d6a` | `f72a22e9f79e89847f95b192bc1504a1a96b25255ae360a0e52b956b86cbaf1a` | `036354c8f4be80397c3c2cbf1fb552a31eb2a1b310bdc5d6e9147daa32f26e5c` |
| `geometria_celula.json` | `02c116803207b295f88dc230a571e6814b7a872929025c1a5e8d11938a58b0b5` | `73dcc872de7c845ce35c2e555c0ac0da92086267dd0866900a993ae94c681d7d` | `15bc7cb38a20f292d257e6e5a0a500d33bcb01e1b0b4b4bc80b92548d6846ad0` |
| `juiz_grade_nativa_diretas.json` | `5d8e47397eac39122d7d9b52e3118491c4547cffa6c856e4477d0fe09c6d76f5` | `9253c10b92cf833a8b41ca89a5e0d9c336e9ac93b01a55c210afb975fa53ce55` | `e93d1a5156c19194404698fafb45a62c848fe01a1f87dafd6323ba0b19027052` |
| `juiz_lidar_v4.json` | `4c02e2edbe0deff2e2edab6078f28443c1fc6c32e5b2b22162ad2096eaac39f3` | `30d680e796b23f19628293f1ea3e5a353585ca99bbd6c61deea4b5355755cf2b` | `f85658b91af25c33a8c2cc3e976592a165527fe2eb3c990f3e8ea15fc0700c62` |
| `medir_delta_escala_v3_nativa.json` | `6c4638c25c95ee538552a20df8a10dfa15906752ec7f2d20ba0db534eec05438` | `f160ac58d0c65a8a2e7c34ba1194c52619fe95c531cf001052c09ddc31e952c2` | `d5907da39a98f261492ccd66c77506eeaa441a48c9d574d35b2cdb301ee75bb6` |
| `results/baseline_mlp_b4s035.json` | `e70c0d3bf5b7eaeb9760fbfe02811de9e7cb8541bb95524dc20879a6faa2ca84` | `9e37ddd06b88281c3f4353adcef715eff4dafe9a763f433be49600a5f0910725` | `19c488849b5b89c3e0c0222ce4408f2e8c04e0f19fc3729b197bc3fb96e73246` |
| `results/d1_orcamento_gatv2_2_semobs.json` | `2b5ac8c07523e7c770536737a89a8766f3da6a5c99a53cca7d28cc83a823b371` | `9a4cfa20d59c571fe28be67e132356ae43066d6869285d05159417ee102cd672` | `84b83eefa6f31fdcf3b74b96349602586d47ffb7406846184e0fbd134dbba69c` |
| `results/d1_orcamento_mlp_semobs.json` | `238f7230fdc406f90e2296363947cf6c1ef5e4e6573596a4d384f721f25466a4` | `d5820c4200132405e8e712e7f88d89b00542d89116317c248ee74ef040eab2d7` | `cfe919b28a905116be66581b41913f82730e59d2946e90a0cd7a58827abcfce0` |
| `results/f1_rampa_gatv2_2_semobs.json` | `b4bdd3fde61eb4eacd9d6434bec38b5c265878abbd20e0e1756e359b7d83b525` | `548c6da7cd10bc5024736c9c2028e2a4f2c32cabba8de74628538fff15153cb2` | `992bd77d84a6d2fe1aba2a3bd66922c188009859493da0eafa87261740eeb20e` |
| `results/f1_rampa_mlp_semobs.json` | `ff09af28791d8a19bc6ffe50ccdc01f3ca37f2fbffb525ac7bd90b9453cb0a6f` | `cccc3d35ba48736da4b2a655cce2ab28f87d039e4be6622273f93ce9f9ce8703` | `a2ef249b6257b6825485f1df3ef6491dadaf2ab0922f52faaa65309ede086c78` |
| `results/f2_arvores.json` | `c5c53ff7cb6c1b251eb25e1ac1ee8d92e4e0547c3085bcaadb41189df9d43db7` | `e637ff1f749554defd8450b76b8218efe6593310ceff2365848e3a14125f354e` | `6c3ea52b8eaba3deeb5d9accb493dff42f088faee7130288e7a1690120aee177` |
| `results/noite_lote1536_gatv2_2_semobs.json` | `c12525b8649cedda44dcb027ba5141383d9af9e47c5302ed219d6d0f1d166399` | `351309399183dac9e2bc735116efe34d589ae24878e99987e690bf3c4e096469` | `6b2fcb96d2fbc23ad3b694c1c00d28d20dee9f82cd0e5396e2f1ee65836fc97e` |
| `results/noite_lote1536_mlp_semobs.json` | `b7aa85db941a22d0f8540a0ae493e6eb297078714546f26b69c41ef7d1e9c2f6` | `7f74a6ef3203fbc8773b0c29bee700ee6f03d18fec74f2b6ef44f73d49b7f7ce` | `46aa32a97074166046d0f759c04a9ab231f3637117836982f0b8347f53b39af8` |
| `results/ofat_boost_ba16.json` | `a50ca9c6595e7b35c2117bf4e07fa3e2b6237746f9d72f2e9e1f553f7268afa9` | `fbab7c54da40624ddf911baead3c594e76214a21ec899d6ee33e729e0364a5d4` | `d2ae4370721333fd9a6f6bc8d53c6dd7afa56b6a23ce2f910380c880782c1bd9` |
| `results/ofat_boost_ba2.json` | `94a5e4fa75fb16ec8b8407e6d151e540022833dab94eece05def7a0c4cc94568` | `ead288a3f222344ce98b5485247af66cebf9f952cf5fbda0ba9d9276accf192f` | `e246b3ce6ab5a8c18f43e520a0f101593abe273c835acbb395fe6f3e4a1c87ba` |
| `results/ofat_boost_ba4.json` | `2b16594822059650225e63e6551a3c26039dbcddd8b1bb1986dc7f466e6ec68f` | `ff7efc3160a60cd3a7c0dd0a5780b923a661a04b341573978353269f67961bca` | `3d3bfeebd3009ec95900dddca2590d69544c743fec4dc3d81fdc63a2bccc0294` |
| `results/ofat_dossel_pd137.json` | `fccf02b417e0e75ab3f344f74768c69af2515f879537a97ba74606f58f346c05` | `cf5c95746681f5dce216ab63ce738b9ee1b079f32441afe30468655f38ee86e6` | `c0df8e1519a5625fb86040a91d88852c689c6020241deddf9ab1b76bfa6c9dc4` |
| `results/ofat_suavfina_b4s025.json` | `1a6fd30f06651f5cf557cb96ba89810719a93baa5a6c7a202c062d99a9d4445b` | `12a3a735ca2e1b79846ae360e8d9540de69e9a75a2683885454ebf95fd3f34bb` | `9495665338ac792172b91548a28d05cf5f89de47a39e665da71d9a7e8c365fbb` |
| `results/ofat_suavfina_b4s035.json` | `345f448bdf52a29ed2f43706724746131cf370b845ad5811b7f83e67938b0bc8` | `fd0c91ac379d6dc336e341c4b93580328861fd66e4460292918b923e984fb232` | `8bea3f5ab0591bc8edf53723d1b113d68e7eecd7b89360ea70d50545ae07d099` |
| `results/ofat_suavfina_b4s05.json` | `07b60dd10a590910b5095782e49144223a5fb61cdceca16026ecbe120b0765a3` | `1b18033f2ebf6233bcbbc90d73234dc64a51f48a5e159a4c2662dcd4a2878f27` | `7f99f88daeec5f60f4679153842035c64d4fda10ab3ba88adc873b06f1b063fb` |
| `selecao_em_validacao.json` | `aaf63499678c74c9edf9e694672c958081bf70c5a84e97507bbe1bc09de2f149` | `be5aab3352b54f1762e2a64bc94daac81f78ea8d6b374972c27b988ddb85fa30` | `93e1619d18477fcfdb01f2dbdc8c033903d60e2567db9f6ec85e026064fd6f36` |
| `treinar_arvore_persistir.json` | `b028e3b32e0479cfecb038286ab5247b85aa0c4f5607188b469b08338c3c8213` | `aaca5093b2e926a8aa8551813dc9f0d8440158913fdae6f0e8ee067f2f20a49c` | `2f29aaf8926a2bb01b83c897fa11886f7968b2585750474386d145a3558e3faf` |
| `treinar_arvore_persistir_v2.json` | `5be410e2c01fd63a5969ce3af2e1c47efdb45fc2705f096c8083b25931911cd9` | `3f7120cd71a9990c926dfb7ad875db7500e137f4b9e423324438554fedef8e3e` | `ce7ff5076bd3581276ceb0d3dd58ba0e0765dad6c218a35763bca58a9ce94b3e` |

## Scripts: comments cleaned, code unchanged

Every Python source had internal working comments removed; comments, docstrings and the string
literals that cited internal review steps were rewritten in English. No executable code changed:
`verify_code.py` recomputes, for each script, the digest of its syntax tree with docstrings dropped
and free-text strings masked, and compares it with the digest of the original (last column, also in
`CODE_AST_SHA256`). The first digest is the script as it stood in the experiment tree, which is the
value a stamp's `sha256_script` refers to when the stamp was written by that version; the second is
the file published here. One literal that the code searches for in a design note was left as is.

| script | sha256 in the experiment tree | sha256 published here | code digest |
|---|---|---|
| `_arquivo_2026-08-09/auditorias_pontuais/qualidade_lidar_4biomas.py` | `386acf85941b8b4ede619353c0f7a165407af9ce543050372a689539550d23eb` | `e949616ebd3a9a35ae7c5ce92c0c6c29b53e63243e382bd1eb508a7bd5d42941` | `2e8f99df4ced5d8b23d7ed7b3489cc4d6ac66bd8d6134411dd24702c514458f3` |
| `_arquivo_2026-08-09/auditorias_pontuais/qualidade_lidar_do_zero.py` | `f20746e3e4b76542d7a5a1c29bc5a21898425d550c3997eb7e248300be235f8f` | `cbf2b89ce89629ea521aeeff09363700d2c15db7d1c9134f4f61de91aff1270c` | `abb3cb16dabf9d89a0ab1c6a134da2df81457e30a8c303dbf6a08902e8d9a079` |
| `auditar_admissibilidade.py` | `862579a606079ecd40f46a3aff1c370821370f54c1b5963b9510cac869cd333c` | `10acea6b2ab278371aab55b7ce067310b98e09b35cc3ffb327cd544d4812290e` | `92636d6ca209a686a806acad1bff0e7cab291248ba3d60d96222039e765ea761` |
| `auditar_d5_invariancia.py` | `1fd917242f00b06afff4f55925eb667d59c963c6bd9971e882d9d5f1577491f9` | `0261314bf82ea5aa4bc29976fa190c44339e044f9ba5eefcde4584bcd7a06acb` | `c366eb92288ebd8314b718fa31be1a29b5e6bc8e15b3303a348412ab109e4bc4` |
| `auditar_d8_triangulo.py` | `20c60c435ea671c4f2b568e44cc1a7ceccd21cec50f7c9ac12d4852f10d697a1` | `2093d9c87a6f0e790eb2064d39f67ad1a92f97cef3e4a86c308de1a3b0386035` | `448663ab9537f0f196fa99b3f56cba946fbc03379a53bfa84efcaa9ce62ab846` |
| `auditar_densidade_solo.py` | `c593d3a0b2c835141732141c39dce74425e0db4961c02e1d8c7d9bbd26dbe316` | `248aee558b7645da48551394aee21d11fb36b20fbd9a2bd6404ac2cd9aa4da79` | `b2f01002532249adaadcd9238f930381bbec27fe4b87a855143dca29c05c720b` |
| `auditar_dt3_criterios_registro.py` | `3e8ab2c311e79a68db380453a9871cffc6085db75ddb5802ae5244173a709535` | `62e1c4cce50662a92935dc3fbe930dc9a9789a2c09b21be398dfababb8e7eb79` | `5bd5db4daacc222cea53050df2f58bd568d0af5a3e2d64131270555cb591d84a` |
| `auditar_f1_rampa.py` | `d83092220e6e99203a47b39697adbdeaebd3b0694d2c3c3ea37420749e70742d` | `ebc5bd540ade4d4c5e0828a709feed1499d465b6055d9d7566913c787be696a1` | `312fbf9bd355145b6db9949ab314ad06eae853961a86ff328ea9c23ad2ad0c29` |
| `auditar_f2_celula.py` | `4e730a46e832a9802a77c9eba7aa812c5ffc79d6442b333524deec2b9269e0bc` | `d250ac4e92988fafeb8effa99f1d2a14ae6ac766523945e272ea647a2eda3bde` | `e1a73a4a9e4cbf848efd7023d47402a05be156206918b442b6cc5a902b27d037` |
| `auditar_lote1536.py` | `d37c3fe13c638ee07f7f7ec0100c0203543518c93f74d04a1e1f0574b92064c6` | `1b1319577be57bcc728753c3734fe507908c119ccc846fbe1f51ae62dcbf35af` | `682c452ee6ee5547975a1479e451c83efd9039d51178dc132a3f144ff3e41819` |
| `auditar_nmad_pareado.py` | `52e471fc4df6e41f1f87ca6de800275770e7bb6eb1b59b6a84bfd9ce0b57a3a1` | `9a4df65ccea53cec3debc12fecfbfeda021ee9a2471813b4ddb2af39ae40364f` | `c27f8bda0b567879bc6c19cff891a6c3936ed9032a9660cd94bb5cbbdb63c093` |
| `auditar_nmad_pareado_nativa_anadem.py` | `f022646ea98e0ae3c4c7de9ce39e73c901a752442c8763e7564654ee115ea9d9` | `774fb0f7411bc10634f30eda8efc164f089c9f12b41e5ff59730bddde3ba56b7` | `c100b3715499fef4f11356bad18cebdcb8cb126f1e969436d7e1d5cb669e835f` |
| `auditar_paridade_capacidade.py` | `313bf2a136cad9ff56840b2377076e4917a47e78572d81678d3d8e8e224c0a4f` | `718d659e1a06540fbf77eefffd80559d24d67ed8046273490a4c99a3d4a45c76` | `5d6042f75d4e1c42385ff61a68de693457a506cd054cb80b280ed7a470465e34` |
| `auditar_perfil_anadem_faixa.py` | `ef6cd0fe52bfd3c232af33f73cfce3e982860f08531b56f442de01ce3e73a0b9` | `041bc7cbbbdc70b83f0436c68831b3d7f1628ce43f3cf3cf5400b715641dd79c` | `b46b7972616ee711c94761ba8cb8703da2c515e9924a8ee06ff941dcf6997e74` |
| `auditar_recursos_s7.py` | `32daadcc9a3e36be73fec020beba8b1e99882dbcf3db17aa09a00ae054089bbc` | `07d26ae9a60c702246768263d2f99f72508d6bb82bbfb671bb5c6759b09acdf9` | `b258a1ecc6963056753e53efd92a4d9c4a1c7f05f5f356e665bf430df1e4dcc0` |
| `auditar_rf_filtro_gedi.py` | `403b52f925dbf1af2a5bc37e9d0eaa1c3ef704abd9d1c01a8072e1895f97fffe` | `faea68a783bf5253ee4eb06877be81f3bcae33430ec8504f140696aa79684cad` | `e86d800012e3e569b50060c46cf77d17daffb31512282f2e8477486d92cfa024` |
| `auditar_rf_filtro_gedi_v2.py` | `d594f0e8e127152ed9526b4a3486fd4ee34763247e971ba97ea46c68cacdfb53` | `24a59b8d08040842de2f37acea1c2cad10edff2309bd8ba46022da2962540e25` | `2b9e5e72abdafbe748111a2b79b685964bd29d0ddd608eb3a27121f6e5e4097f` |
| `auditar_robustez_o3.py` | `c05944eeda4d6d65082bd9c1db55957968eb189906b0bc2cebbac5905041a1a9` | `b105226e5ab4ecfec438afa472c26d41e6b48c29abfb7d2f9e175b5c7075408d` | `7356ea0a59f5689b65bedb7a3bb2cae1eca54d1674140a98d24ef354373c0518` |
| `auditar_rotulo_regua_v2.py` | `f50f622cbcdd5aab1c05d8e7682e87937aaf18e8cea16f6f5bb5833a685257d2` | `ed720086fb3b2980272bed67823bb6504393adade483290758e80e99825412ec` | `612db249bfcfcb0d00b7ac497e7447c1a7efa15dbbcd9d85642334c48da4efc0` |
| `auditar_s1_estratificacao_lidar_v2.py` | `836ff7b447671a7f8b005ba2d8c5397f6526693b48981b9ad859e9e82b30a9d6` | `9c37f79e7d3118e8da640f10421ebdfc21cab65a7e628083a5905a082a85307d` | `30023c0ef1fe641e765a377797b5befdcb071a729c4b599b59faef59c65168c8` |
| `auditar_s2_registro_referencia.py` | `72041a6e6f30f7c35fd7781d9c45ada59153edab0486e1b5505dc65a0cef5161` | `d9580b167cb8f019dff1705c3d5ef5ebdefef4a39f09bf0be121cf2451049ecf` | `bd65f8d78ece91eadf3e4e612aa8cee259b88fdbb35bf6d50115f038fddb03a4` |
| `auditar_s5_baseline_fisico.py` | `a66c8c38834ed9e93e801c225ddc7c54d1277e86b1d69cf9ece68296909f9e0b` | `bcec3649f09051d4c0ccba813a24e7b49a9cee8d252a89fcf5074973f043da5e` | `9fcc546b71f6aa5bec21ce0026bd7f5ef9c40f0c9e31aa0bd8860c6d3892b627` |
| `auditar_s7_arvore_lidar.py` | `915923e62226d6a7cd957e5b387518b1fb49c9e2f6a872fb842c225a603928ab` | `31353eef19e779be50df37f8f9f786ab5baffae5fe3c95da8b2455cef1ad5307` | `0b60e7ba14601b761ff2c7ba809fb77fcd5a65dd343b30c4b3249d335209507f` |
| `auditar_s7_arvore_lidar_v2.py` | `3966e5c797fa5993b75ac26ab6f35a05b9472ddbd62d14629aff8893886753b1` | `026b119ecca775589d5917bac22e6230eb7e135ac907a0ab200e767d4db0e23b` | `8697f1e98aa1c3aef1b1764074814b477a8e3424dc6d947569597bcb5c7ad9b3` |
| `auditar_s8_mascara_disturbio.py` | `3c27b9834061941bd91e4ed103b91d61e7a08a533e6583a42b812a4d4f301a1a` | `d4068adf2452caa945c50c648569b93ca77be00fae0d2a738ee60b473675606c` | `3cfd5434f20f6088ba929909b6ae3bfa228e18e1007be741c99182ee01598434` |
| `auditar_tabela3.py` | `c7691eab1b1d77d28d8bec26a54207dcfe87352f062c1a0f77809b9a4c719d52` | `369bd1b8962e1de4174610d89505dc381653c4426869757e1b5cc6d88ac2b849` | `97ab89a9db59f4956eaf9b0740a47519c9afd45c5e518d7bf8dc27b977caf41d` |
| `auditar_unidade_quadrante.py` | `8864dc12ca288bb8dbba9f59a8e6a5c9b3e0d4c01ae0889cac198eb7a189172f` | `c1e43fdbd281b644a566334c515fe4e38a25028c18b0d7212b1040b4fd39a2a4` | `157ce37199feb4e8c2912d680b2c3bf5617a69ba2848abc86e2b24df7747a392` |
| `b2_transferencia.py` | `aad40bb59cc7c5bf57dbc0ac789434fedc029803b97ebe445f48d6a95749d389` | `1d3a4c17d3a065086dc581b3f1e85f72b7e1316ac63fb0ade550e920c2254b25` | `cdec55f28dec50f59ef07c492bcef8d3cdad41aed8c8d1a10e74c8e07f2bfaac` |
| `braco_arvores.py` | `8f3b1544535d55c653944caf06cc6bedf193ce0d3c7ed219d5f46a76030416b2` | `fbd26fcc89076dbbc9fbdf914f47fc054d0a01d819acc68f3b23c1b2f9e352b2` | `af87bdec4b4a167fba4faf961da9d528066cedc779d0cbdad858c3dece5fa6a3` |
| `fig/gerar_fig1.py` | `7f2c5f88cabbd34b4beecba170c0bb33f5061637a480300f242a151efc45d87d` | `616a19faf99415e6ae439e5683c73f5b8c0b8198a3b25abaf54ce7c78da31d78` | `8a3a31f0705c6a865a79668ad0edc0cd9d632701012f8a794626cb3c6a6fa09e` |
| `fig/gerar_fig3.py` | `9c486c10575ec13164c66f76ed625675447c440f447ffaacbad6eac449aafd09` | `7bbbcdd655fbb3a638c4b2eb91fa8e6ffc11cdfd8ed80060cb2a561cc1b14185` | `b99fb0338176708011deeeb735f2bb5bef3cef32001ba18ccdd6259174ba3f83` |
| `fig/gerar_fig8.py` | `b4ec286e2ba341638d96f84fb21ac4532e16a4745f80d927e55654e4a88a299f` | `23bf293af3deef116fddaaf66c10f5cbedc2a14ad7f54dca44f64a5cd5c76641` | `41d54164920d0c879e73dd8390ce42cb73c001e56295632d517056b690b1d73e` |
| `folha_de_fatos_acdsa.py` | `3218e5e3a720ee3a82f35094444a6f0b155103b0c331596c6c8e47197a5efc51` | `33262f63c277b643f662d5062c49e52833421492f6a67c42158ad58c6dfd42f7` | `829bb53f579c4772f5d703cf2c6065efca38fc07cd221f2361ca4c594b4271fa` |
| `folha_de_fatos_jstars.py` | `add0cee5457b6c9443d9f543da32a74f154a79b3a98c9df917ad0aa196e44c5f` | `2c190909336a0c329956b39fe9948bb52b9acf06cfe799122bc1ab9ccaa09339` | `21c2da03f727672098336c4656b424a6f83e72ca2e306074402b51f9a4aca221` |
| `folha_de_fatos_jstars_v2.py` | `d37ab07baa4037e75b5fd9158822c0186b1264f7dd70b409446a439d7ae00160` | `f33c2de8c150154dab32d5b9ef17e27ec7c1716d8a591aa1f792b5e7148e1248` | `de5fe1697299f5c0641b244d0ba2d5ae37297e97fb8a017a4a3ed07803321388` |
| `folha_de_fatos_jstars_v21.py` | `d8a2db68fc2b9f00eba41355da6f021d9fed7aef3ac40fa6d3c94450fa19249d` | `863ac5984115671fa994843b0fc3399c9d1ab1b5d0e3d6f0a564b4d15cbcda05` | `cb87fb5c5cbc605e8da8a202b78997b0b31aceb8fec66f6726206f685cc7ea82` |
| `folha_de_fatos_jstars_v22.py` | `aacb1b41c1642aa85646d71aa9b27d17599c3dafd1c26e44bf9377b32cb675cc` | `613bbe054f73ffa096157d656d20c86a4182a0decf1e85b331412f7210ca20f6` | `3f909cd7b3a2ea459d37679128d450c71119174424fc7deddebb49179d1b8b91` |
| `folha_de_fatos_jstars_v23.py` | `a120ddcb6baf335cc71d991849bed3674ba62d318fc0b959cbb928e76f178e5f` | `b024cf8afc6d2413f7a0c63044df325ddc5ed6ce8f6def6c5955473e472a1891` | `67c1b37b0f99029a81d0981ad176bfb6f605d00a9840fa636af82b421c9f1aaa` |
| `folha_de_fatos_jstars_v24.py` | `9d24b51bcb2ef2d71a2cd9445d5c0a8473d63fcfe45773af7e1c1576a45ddd37` | `16469ae9dd67f4eae151fe3cab5760b38e20df51c8723fea1eec1d9bea37832b` | `1164b3f7b8ce08b66d73736d658e05123f89a88708228050b091dc93f2730b2c` |
| `folha_de_fatos_jstars_v25.py` | `ed8b234ca16e6e8a880f87588af807b5d8b9f405a833a9396a8a2b82e7b159e7` | `33ba6df1e9f14684495713cec22935e934bd755cbf2df881a011311acd14770a` | `fd725f2add47c8407d6c251f48b114ad335073c68d2bf31ea8930b83aa03cce7` |
| `folha_de_fatos_jstars_v26.py` | `cd8372da7945760632aeb487fdd431fddad53ea925e0d73f51b4bcb3b14b6379` | `95c81b2421c913180e686d87747b18e0a5be2b6ce064ebee5c53b776100f8157` | `c975ad3b3aa7f08ab0548709f6a73148d9740707f3462ad0c2ff6ceb74b444bf` |
| `folha_de_fatos_jstars_v27.py` | `1600e05e36f3dae089fe7824f2233bd63559bf841381dc09f89ec27704e511d4` | `b3a8bfa62f1da324ea57ae7ee69af7d75602dee631d10ec9b282008ee073b0cb` | `066a0b22bf67cd84dd49a0d0214ef4a0d55398414a3c6c27b026db762bbc080a` |
| `geoide_v23.py` | `001184fd628bba0c97fa030a20292026119260fec8cd21a18f3964aa49001796` | `daaa70cf6c0900e753638ba2361bb140d9380e7b9316972100a3a7dda79b7e43` | `8139090341bb35e1bf4efe1d5a769d638b69ab1f3a7a4ea7cf87f2a36b5e52a6` |
| `geometria_celula.py` | `a415a6d335298918a9c57dfb8798234b7fdcadef73a01e1efe22f4159c0fc6a3` | `04b2117bb4971766638aed8f9818f40a2e6d3ad42b300d12a80244d0be7f8649` | `7d33dfa08d484613834ae277b2a7a36e8b6b4cda803d9bab27b4cc49c6d50c38` |
| `grade_v23.py` | `b5765a4047899145075b69dbe856a75421cd363a74e2fbb4426d913368f4d2df` | `840f72d6457b146e9cf7b3332a3b70e95df22caf4afb3e4733c753ce34086946` | `a6f7cd77f6dcb617115710bd81dbffefee1f367ca3e803eab2ae0097349e5a82` |
| `juiz_lidar_v4.py` | `7768ba350a38cc94e9ab37a48cbac1b7b8917a902d3eb015ad430919c8b033c2` | `e94d6eb070395413c29da1d671d0ff3e69c87b324f1e1cf5d41ea386a4f371dd` | `55f78f11553cad0432b1d8bb0101c3acf81ade80d62831db4b257f928e86f2ab` |
| `medir_delta_escala_v3_nativa.py` | `c122a103c92af765261f33bf885e16723b110fe2fb2cd1e0b64a2eb208960439` | `4d56ece1729e96db2bcf60294512347a66dab99ef9a4055c4d37f84e325f1ae6` | `e05b2e5bf1904a2d6116ae3242ccacd00e966290d550324677d274994222732c` |
| `preparar_laterais_nativa_v3_diretas.py` | `b52e1cf6fff507020ca5d3e7acab2a3ae5c599c807acc074196c506631a76cd9` | `2c8cb0af1e1403190d33c9062dceda955caf0626dc1899c650cdb9d5228f2c87` | `ca73527f101b0c876e108b7b1974acae5b2a18e18180846384d7872d08bf2195` |
| `proveniencia.py` | `3f5716c109e53039ef1c74c345f5935fed4581278789b690c7993be3a86918fe` | `8e0701cd1e7f647399c4eaa38de8cd49a12405c09601034e68a4900da26b5f3c` | `5459b510ab66d00075f42013ef6945d43bf8b6b7ed6da59ba1afda7f53e73ad0` |
| `regime.py` | `b7e68338e02c5e4c602f68def091e1e63f0842963d1b4f594f37091156423566` | `7b44052af58a9b5babfa49cf69cab4f53e69d2bd7687ceda09a74aebd444900c` | `6b7f7969eea982bc9158f87f0549cf2f71b8d8ed8195a1d30df2ef9be93281b5` |
| `rotulos_lidar_v23.py` | `bf8e36cfd2281b2eefae3cdd36b8b8b51cf145bc6a7e808008cc574792e14f8d` | `7c887e5c3503dafcc3a30ef61989412bdd7898fb7b7eb13e0b4f44b05f58893f` | `5793479d0a4f247015e8063e893afea9c9d69008951700b894a048f50d97f04f` |
| `selecao_em_validacao.py` | `bac2f4d620a3e8285e1d0af5339ece67dc014f2aadd29e7654ece0fe2d18b84d` | `ddc2fbab0dbe25aa7acbf8c8e7e9218ebc1038224ee2d1eee99b36cdf7f27843` | `2bf161fba89ddd7e2ccdde09b851ecfdd13c2a7db2f1bad8762547922104cdd9` |
| `topo_v23.py` | `9b85234dc6a4ea3b629f5268c25fd6dc5ce29cf4b809c1a0567523b482e44507` | `53a7a2a0d4b9aeb69176f021b8765c0e479e570d623815a425f403f45da05f8c` | `15d4637869c150a1d3f31d8f17205c20c96194556292a3f69b5060ad4c72173e` |
| `treinar_arvore_persistir.py` | `6bdf215c5c0e5253c713050842edc1381443af1a8e1d2f4b054de6a3327a3ad8` | `53a4d974059a131fcff7e144f9aa8823b58fb60af22ef822f2ccfc3840180e8c` | `e80c5717bcd62820379a67c7274915389e35bd1e53310a83faf1ff483f5bb8ac` |
| `treinar_arvore_persistir_v2.py` | `9a4b57a0490103aa840bc6a6ef0e10ff90e25eb17a7b478beb00b07f15296ed6` | `33dcc2bb05250b21130bd647847b30a5adcbbc8181b3fa2a674c9daeafeac7f2` | `0c8f12caaab5ab944b9fef2cc0c9c24e66e7927253030d0339427abf070357bd` |
| `vetor_v23.py` | `743e6e3a76580d5362b6c21e2c623fba76f8b2b305563845a6bcef78348f1985` | `ca1f7a928120b1bf07313ece1771de67bc79b58f1e315b9fd4f91021b9816997` | `2b621e6e87c914052d20e3e0269b79f701b39bc44f6e8b2c7c84a3373cef4c1b` |

## Artifacts without a recorded generating script

The following artifacts carry no `_proveniencia.script` field. Most were produced before the
provenance convention was adopted; the notes say what is known about each.

- `_arquivo_2026-08-09/auditorias_pontuais/qualidade_lidar_4biomas.json`: early profile of GEDI quality fields against the input surface over exposed soil, written by the script of the same name; descriptive, not used for any number in the article.
- `_arquivo_2026-08-09/auditorias_pontuais/qualidade_lidar_do_zero.json`: early profile of GEDI quality fields in one savanna quadrant, written by the script of the same name; descriptive, not used for any number in the article.
- `_pareceres_2026-09-28_execucao/geo_produto_anadem.json`: ANADEM vertical-datum test; generating script not recovered.
- `_pareceres_2026-09-28_execucao/guarda_sanidade_anadem_vs_glo30.json`: ANADEM vs GLO-30 sanity check in canopy gaps; generating script not recovered.
- `comparar_ponto_s035.json`: stamped retroactively; the script that produced it is no longer in the experiment tree, so this number cannot be regenerated from code.
- `comparar_suavidade_fina_ba4.json`: no `_proveniencia` and no `_fontes`; the generating script is no longer in the experiment tree.
- `f2_arvores_config.json`: tree-ensemble configuration fixed by the earlier hyperparameter search in `braco_arvores.py` (recorded in `results/f2_arvores.json`); read, not searched again, by the tree-training scripts.
- `fig/fig1_numeros.json`: written by `fig/gerar_fig1.py` (its `fontes` field lists the same inputs the script reads).
- `fig/fig3_numeros.json`: numeric sidecar of the three-families figure, written by `fig/gerar_fig3.py`.
- `fig/fig9_numeros.json`: numeric sidecar of a figure panel that was cut from the submitted manuscript; its generator is not in the experiment tree. It is kept because an earlier fact-sheet version reads it.
- `results/d1_orcamento_gatv2_2_semobs.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/d1_orcamento_mlp_semobs.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/f1_rampa_gatv2_2_semobs.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/f1_rampa_mlp_semobs.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/noite_lote1536_gatv2_2_semobs.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/noite_lote1536_mlp_semobs.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/ofat_boost_ba16.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/ofat_boost_ba2.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/ofat_boost_ba4.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/ofat_dossel_pd137.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/ofat_suavfina_b4s025.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.
- `results/ofat_suavfina_b4s05.json`: training/evaluation output with the same schema as the outputs stamped by `b2_transferencia.py`; attributed to that script by schema, not by a stamp.

## Other known limits

- `treinar_arvore_persistir.py` and `treinar_arvore_persistir_v2.py` record in their stamps that the
  working tree had uncommitted changes when they ran, so the exact source of those runs is not
  pinned by a commit. The persisted predictions they produced are pinned by sha256 in
  `treinar_arvore_persistir_v2.json`.
- `folha_de_fatos_acdsa.py` / `.json` belong to a companion conference paper; the JSTARS fact sheet
  reads two of its facts. Its own audit chain is not reproduced here.
- Not distributed (the scripts document the expected paths): the raster input stack, the ETH
  canopy-height and lidar label arrays (`SATELITES/laterais/*.npz`), per-cell prediction arrays,
  and the five XGBoost checkpoints `arvore_f2_s7_seed{42,123,7,2024,31}.ubj`, whose sha256 are
  recorded in `treinar_arvore_persistir_v2.json`.
- The label-filter check (`auditar_rf_filtro_gedi.py` and its `_v2` complement) and the strip profile
  (`auditar_perfil_anadem_faixa.py`) read the GEDI and ICESat-2 shot tables, the GLO-30 raster and the
  per-quadrant arrays, none of which is distributed here.
- `auditar_s7_arvore_lidar.py` is the first version of the tree-ensemble audit; the stamped artifact
  `auditar_s7_arvore_lidar.json` was written by `auditar_s7_arvore_lidar_v2.py`. The first version is
  included because the latest fact sheet reads its source.
- The fact-sheet builders also read the project's internal design and review notes, which are not
  distributed; their sha256 appear in the fact sheet's `_fontes`.
