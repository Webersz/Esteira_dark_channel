# CONTEXT — dark-factory-os

Documento de contexto do projeto: o que é, como foi construído, decisões tomadas, o que já foi
validado, problemas conhecidos e próximos passos. Serve para retomar o trabalho (por uma pessoa
ou por um agente de IA) sem precisar reler o histórico das conversas.

> Última atualização: 26/10/2026 · Repositório: https://github.com/Webersz/Esteira_dark_channel

---

## 1. Visão geral

**Objetivo:** fábrica 100% autônoma de vídeos curtos "dark channel" (sem rosto) para
**5 canais**, publicando **1 vídeo por canal por dia** no **YouTube Shorts** e no **TikTok**.

**Requisitos do produto:**
- Duração de **62 a 68 s** (elegibilidade de monetização).
- **16 cenas** de ~3,8–4,2 s (ritmo de alta retenção, nada de slideshow parado).
- Legenda karaokê palavra a palavra queimada no vídeo.
- Publicação automática + comentário de engajamento.
- **Custo marginal zero** (só APIs gratuitas e ferramentas open source).

**Especificações de origem** (fornecidas pelo dono do projeto, coladas no chat):
1. `SPEC` — arquitetura base (árvore de pastas, stack, regras rígidas).
2. `SPEC_UPGRADES` — engenharia de retenção, fallbacks, suíte de autenticação, analytics, CLI.

**Regras rígidas da spec (respeitadas):**
- Nada de MoviePy: FFmpeg/ffprobe puros via `subprocess`.
- Só o SDK novo `google-genai` (`from google import genai`), modelo `gemini-2.5-flash` com
  Structured Outputs (`response_schema`). Nunca o legado `google-generativeai`.
- Python 3.11+, PostgreSQL + Celery + Redis.

---

## 2. Os 5 canais (`config/channels_config.json`)

| ID | Nome | Nicho | Estilo | Voz | Trilha | Horário |
|---|---|---|---|---|---|---|
| `nicho_1` | Arquivos Sombrios | Mistérios reais, desaparecimentos | cinematic + B-roll | Antonio, −2 st | dark_tension | 11:30 |
| `nicho_2` | Bizarro 3D | Curiosidades do corpo/ciência | **3D + mascote raposa** | Francisca, +1,5 st | quirky_upbeat | 12:30 |
| `nicho_3` | Mente Oculta | Psicologia e vieses cognitivos | cinematic + B-roll | Antonio, −1 st | dark_tension | 18:00 |
| `nicho_4` | Bichos Absurdos | Animais e natureza | **3D + mascote sapo** | Antonio, +2,5 st | quirky_upbeat | 15:00 |
| `nicho_5` | Contos da Madrugada | Micro-contos de terror (ficção) | cinematic | Antonio, −3 st | dark_tension | 21:30 |

- Idioma: pt-BR em todos.
- A produção começa ~75 min antes do horário de postagem.
- Os nomes, nichos e horários são sugestões iniciais; ajuste livremente no JSON.

---

## 3. Arquitetura

```
Celery Beat ──► produce_video (fila render, concorrência 1)
                 │
                 ├─ 1. script_generator  Gemini 2.5 Flash → 16 cenas validadas + anti-duplicação
                 ├─ 2. audio_tts         edge-tts → pitch/tempo FFmpeg → janela 62–68 s
                 ├─ 3. subtitles_ass     faster-whisper (tempos) + texto do roteiro → .ass karaokê
                 ├─ 4. media_fetcher     Pollinations (3 níveis) + Pexels opcional + mascote fixo
                 └─ 5. renderer          16 clipes → concat → overlay + legenda + mix de áudio
                 │
                 ▼  (se DRY_RUN=false)
        publish_youtube / publish_tiktok (fila publish, concorrência 1)
                 │
                 ▼
        PostgreSQL (videos_history) · Telegram (alertas) · limpeza de disco
```

### Mapa de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `main_cli.py` | CLI operacional (init, auth, test-render, run-channel, analytics, mascotes) |
| `generate_sfx.py` | Sintetiza SFX, trilhas e overlay de poeira com FFmpeg `lavfi` |
| `auth_youtube.py` | OAuth2 interativo por canal (3 escopos) |
| `auth_tiktok.py` | Login manual no TikTok com perfil persistente + `storage_state.json` |
| `src/config_loader.py` | `.env` + matriz de canais (pydantic, com `defaults` mesclados) |
| `src/database.py` | Tabela `videos_history`, anti-duplicação, status por plataforma |
| `src/alerts.py` | Telegram (alertas e relatório em bloco monoespaçado) |
| `src/tasks.py` | Celery: tarefas, beat schedule, pipeline, publicação, limpeza |
| `src/analytics_reporter.py` | YouTube Analytics v2 (7 dias) + diagnóstico de retenção |
| `src/engine/schemas.py` | Modelos pydantic usados como `response_schema` do Gemini |
| `src/engine/script_generator.py` | Prompt, ganchos, regras de loop/twist, validação, retries |
| `src/engine/audio_tts.py` | edge-tts + `asetrate`/`aresample`/`atempo` + corte de silêncio |
| `src/engine/subtitles_ass.py` | Whisper, alinhamento com o roteiro, gerador `.ass` |
| `src/engine/media_fetcher.py` | Download assíncrono, fallbacks, ritmo grátis, rembg, B-roll |
| `src/engine/mascot.py` | Biblioteca fixa de poses por canal (escolha, preparo, prévia) |
| `src/engine/renderer.py` | Pipeline FFmpeg (zoompan, 2.5D, SFX, ducking, burn-in) |
| `src/publishers/youtube_api.py` | Upload resumível, refresh de token, comentário |
| `src/publishers/tiktok_playwright.py` | Upload via TikTok Studio (Playwright) |

---

## 4. O que foi construído (por fase)

### Fase 1 — Base (commit `4910919`)
- Árvore completa da spec, Docker Compose (postgres, redis, bootstrap, worker-render,
  worker-publish, beat), `env.example`.
- Pipeline completo: roteiro → TTS → Whisper → imagens → render → publicação.
- Render: whip-zoom de 0,23 s em cada corte, Ken Burns + shake matemático em `zoompan`, cenas 2.5D
  (fundo desfocado, sprite com balanço/bob/sombra/entrada de baixo), poeira em blend `screen`,
  color grade, legenda Montserrat ExtraBold com a palavra ativa em amarelo e "pop".
- Áudio: SFX em cada corte (whoosh/impact/pop), trilha com `sidechaincompress` (abaixa sob a voz).

### Fase 2 — Upgrades da SPEC_UPGRADES (commit `d4fe75e`)
- **Ganchos rotativos:** `IN_MEDIA_RES`, `MITO_QUEBRADO`, `NUMERO_ESPECIFICO`, `CONFRONTO_DIRETO`
  (sorteio por vídeo, guardado em `hook_archetype`).
- **Twist na Cena 7:** pedido no prompt e forçado no código (`impact` + `shake`).
- **Loop infinito:** Cena 16 termina com conector aberto que emenda na Cena 1; sem fade-out longo;
  silêncio final cortado.
- **Fallback de imagens em 3 níveis:** Flux ×3 (2/5/10 s) → Turbo ×2 (seed mutada) → frame
  procedural local via FFmpeg. O job diário nunca falha por causa de imagem.
- `auth_youtube.py`, `auth_tiktok.py`, proxy via `TIKTOK_PROXY_*`.
- `analytics_reporter.py` + tarefa semanal (domingo 20:00) + `reports/analytics_YYYY_MM_DD.md`.
- `main_cli.py` com todos os comandos; tabela renomeada para `videos_history`; limpeza de
  intermediários após publicar.

### Fase 3 — Correções dos testes reais (commit `64c492e`)
- Pollinations mudou o plano grátis → ritmo de 1 imagem/46 s sem token (ver §6).
- Acentuação: exemplos do prompt sem acento faziam o Gemini escrever sem acento → TTS pronunciava
  errado. Corrigido + regra explícita de acentuação.
- Legenda passou a usar a grafia do roteiro com os tempos do Whisper (alinhamento `difflib`).
- `.dockerignore` (credenciais, `.env` e `.git` fora da imagem).

### Fase 4 — Mascote fixo por canal (commit `6197b05`)
- Biblioteca `media_assets/characters/<canal>/<pose>.png`, gerada uma vez e reutilizada.
- O Gemini escolhe entre as chaves de pose existentes na pasta.
- Recorte limpa resíduos do rembg; sprite espelhado do lado esquerdo para variar.
- Bibliotecas revisadas: `nicho_2` (4 poses da raposa), `nicho_4` (6 poses do sapo).
- Gemini: 7 tentativas com backoff até 240 s.

### Fase 5 — Documentação (commit `0f67680` + este arquivo)
- `README.md` (uso) e `CONTEXT.md` (contexto completo).

---

## 5. Decisões e desvios da spec (com motivo)

| Tema | Spec pedia | O que foi feito | Por quê |
|---|---|---|---|
| Anti-detect TikTok | ocultar `webdriver`, `AutomationControlled`, digitação "humanizada" | **Não implementado** | Burlar detecção de bot da plataforma. Mantidos: perfil persistente e proxy opcional. |
| Comentário fixado | "pinned comment" | Comentário postado, **sem fixar** | A YouTube Data API não tem endpoint para fixar. |
| Corte de silêncio | `silenceremove=stop_periods=-1` | `areverse → silenceremove → areverse` | O filtro da spec apaga todas as pausas no meio da fala. |
| Fórmulas de SFX | fórmulas simples (senos + ruído) | Síntese mais rica, também via `lavfi` | Mesma exigência (gerar se faltar), resultado melhor. |
| Campos do schema | `numero_cena`, `fala`, `sfx_type`, `intensidade_camera` | posição, `narration`, `sfx`, `camera` | Equivalentes; `hook_shake` → `shake`. |
| Personagem 3D | sprite gerado por cena | Biblioteca fixa por canal | O Flux não mantém a identidade só com a seed. |
| Dados de teste | — | `--test-render` funciona sem Postgres | Facilita testar localmente. |

---

## 6. Lições aprendidas nos testes reais

1. **Pollinations (anônimo) = 1 imagem a cada 45 s por IP.** Fora disso, retorna
   `HTTP 402 FREE_LIMIT`. A janela conta a partir do **fim** da geração anterior, e um timeout
   também conta. Implementado: fila serial + 46 s após o término + uma janela extra após falha.
   Com `POLLINATIONS_TOKEN` (chave "pollen") o ritmo é desligado.
2. **O Gemini copia o estilo dos exemplos.** Exemplos sem acento → roteiro sem acento.
3. **Whisper erra palavras** ("Olhe" → "Oi"). A legenda usa o texto do roteiro e só os tempos do Whisper.
4. **O Flux não mantém personagem** entre gerações, mesmo com a mesma seed e descrição.
   Mesmo dentro da biblioteca, algumas poses saem diferentes: precisa de curadoria humana.
5. **O Gemini 2.5 Flash tem picos de 503 "high demand"** de 15–30 min, mesmo respondendo a
   chamadas pequenas. Os outros modelos (Flash-Lite e Pro) dão **404** com a chave atual, então
   não servem de reserva. **Mitigação:** cascata `GEMINI_MODELS` × `GEMINI_API_KEYS` (uma chave por
   projeto do Google Cloud, pois a cota é por projeto) que pula 404/429/503 na hora, e depois fallback
   gratuito OpenAI-compatível (Groq / GitHub Models / OpenRouter) validado pelo mesmo `VideoScript`.
6. **Perfil do Chromium criado no Windows não abre no Linux** (cookies cifrados pelo SO): por isso
   existe o `storage_state.json`, reimportado no container.
7. **Senha do Postgres com caractere especial** precisa ser codificada na `DATABASE_URL`.

---

## 7. Status atual

### ✅ Validado de verdade (Docker + chaves reais)
- Pipeline completo até o MP4 final (`--test-render`):
  - `nicho_1`: "O Que Aconteceu no Farol das Ilhas Flannan?", 65,9 s, 16 imagens reais, 22 min.
  - `nicho_2`: "Seu estômago derrete metal e você nem sabia!", 64,1 s, 8 cenas com mascote, 25 min.
  - Re-render do `nicho_2` com a biblioteca fixa: mesma raposa em todas as cenas.
- Ganchos, twist na Cena 7, loop na Cena 16, acentuação, alinhamento de legenda.
- Ritmo da Pollinations (zero 402 após o ajuste), fallback procedural, rembg.
- Bootstrap (`--init-assets`) e criação da tabela `videos_history`.

### ⚠️ Escrito, mas nunca executado de verdade
- Upload no YouTube + comentário (precisa de `--auth-youtube`).
- Upload no TikTok (precisa de `--auth-tiktok`; seletores da interface podem ter mudado).
- Relatório do YouTube Analytics (precisa do escopo `yt-analytics.readonly` no token).
- B-roll do Pexels (`PEXELS_API_KEY` provavelmente vazia).
- Alertas do Telegram.
- Workers do Celery + Beat rodando sozinhos (bloqueado pelo item 1 de §8).
- Roteiro novo usando as chaves de pose (o Gemini estava em 503 no dia do teste).

### Infra no momento
- Containers `postgres` e `redis` no ar. Workers e beat **parados**.
- Nenhum vídeo publicado até agora.

---

## 8. Pendências (bloqueiam a produção)

1. **Corrigir o `.env`:** a senha dentro de `DATABASE_URL` não bate com `POSTGRES_PASSWORD`.
   Corrija a **URL**, não a `POSTGRES_PASSWORD` (o banco já foi criado com ela). Caracteres especiais:
   `@`→`%40`, `#`→`%23`, `:`→`%3A`, `/`→`%2F`, `%`→`%25`, `?`→`%3F`, `&`→`%26`.
   Enquanto isso, os testes usam o script temporário `output_test/run_with_db.py` (fora do Git).
2. **Autorizar as contas** (no PC, com navegador):
   - Criar um OAuth Client "Desktop app" no Google Cloud com YouTube Data API v3 e
     YouTube Analytics API ativadas, e salvar como `credentials/client_secret.json`.
   - `python main_cli.py --auth-youtube` e `python main_cli.py --auth-tiktok`.
3. **Primeiro teste de publicação:** `python main_cli.py --run-channel nicho_1`.

---

## 9. Plano / próximos passos

### Curto prazo (colocar no ar)
- [ ] Corrigir o `.env` e subir `worker-render`, `worker-publish` e `beat` com `DRY_RUN=true`.
- [ ] Revisar alguns dias de vídeos gerados automaticamente antes de publicar.
- [ ] Autorizar YouTube/TikTok e validar o primeiro upload de cada plataforma.
- [ ] Testar o TikTok com `TIKTOK_HEADLESS=false`; o headless tende a ser detectado.
- [ ] Configurar Telegram para alertas.
- [ ] Testar `--test-render` para `nicho_3`, `nicho_4` e `nicho_5` (ainda não rodaram).

### Médio prazo (qualidade e custo de tempo)
- [ ] Chave "pollen" da Pollinations: vídeo cai de 13–25 min para ~5 min.
- [ ] `PEXELS_API_KEY` para o B-roll híbrido (`nicho_1`, `nicho_3`).
- [ ] Curadoria dos mascotes: trocar por PNGs próprios com poses mais expressivas
      (o Flux gera expressões muito parecidas).
- [ ] Avaliar a **Content Posting API oficial do TikTok** (exige app aprovado) no lugar do
      Playwright: mais estável e dentro das regras da plataforma.
- [ ] Hospedagem 24/7: VM ARM grátis da Oracle Cloud (sugestão da spec) ou outro servidor.
      Observação: o limite da Pollinations é por IP; todos os canais no mesmo IP dividem a fila.

### Longo prazo (crescimento)
- [ ] Usar o relatório semanal para ajustar ganchos e o twist da Cena 7 automaticamente.
- [ ] Testes A/B de título/gancho por canal.
- [ ] Revisar e trocar nichos com baixo desempenho após algumas semanas de dados.
- [ ] Painel simples (Streamlit ou similar) lendo `videos_history`.

---

## 10. Operação rápida

```bash
docker compose up -d --build                                   # tudo
docker compose run --rm bootstrap python main_cli.py --test-render nicho_1
docker compose logs -f worker-render                           # acompanhar produção
python main_cli.py --init-mascots --channel nicho_2 --force    # refazer biblioteca de mascote
```

- Vídeos de teste: `output_test/<canal>/<timestamp>/final.mp4` (+ `script.json`, `subs.ass`).
- Produção: `output/<canal>/<timestamp>/`; intermediários apagados após publicar e pastas inteiras
  após `KEEP_DAYS`.
- Falhas do TikTok salvam `tiktok_error.png` e `.html` na pasta do vídeo.

### Fora do Git (de propósito)
`.env`, `credentials/` (tokens e perfis), `output/`, `output_test/`, `reports/`,
`media_assets/characters/*/_cache` e `_raw`.
