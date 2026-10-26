# dark-factory-os

Fábrica autônoma de vídeos curtos (YouTube Shorts + TikTok) para 5 canais: 1 vídeo por canal por dia,
62–68 s, 16 cenas, legenda karaokê, publicação automática e comentário de engajamento.

**Stack (custo zero):** Gemini 2.5 Flash (`google-genai`, Structured Outputs) · Pollinations Flux ·
Pexels (opcional) · edge-tts + FFmpeg · faster-whisper · rembg · FFmpeg puro (sem MoviePy) ·
PostgreSQL + Celery + Redis · YouTube Data/Analytics API · Playwright.

---

## Pipeline

1. **Roteiro** (`src/engine/script_generator.py`): 16 cenas validadas, anti-duplicação de tópicos
   (hash + similaridade), gancho sorteado entre 4 arquétipos (`IN_MEDIA_RES`, `MITO_QUEBRADO`,
   `NUMERO_ESPECIFICO`, `CONFRONTO_DIRETO`), reviravolta obrigatória na Cena 7 (impact + shake) e
   Cena 16 emendando gramaticalmente na Cena 1 (loop infinito).
2. **Narração** (`audio_tts.py`): edge-tts + pitch/tempo via FFmpeg, ajuste automático para caber na
   janela de 62–68 s, corte do silêncio final para o loop.
3. **Legendas** (`subtitles_ass.py`): tempos do Whisper + grafia do roteiro (alinhamento difflib),
   `.ass` com destaque palavra a palavra.
4. **Mídia** (`media_fetcher.py`): Pollinations com fallback em 3 níveis (Flux ×3 → Turbo ×2 → frame
   procedural local), ritmo automático para o plano grátis (1 imagem / 45 s por IP), B-roll Pexels
   opcional.
5. **Mascote fixo** (`mascot.py`): biblioteca de poses por canal em `media_assets/characters/<canal>/`,
   gerada uma vez e reutilizada em todos os vídeos.
6. **Render** (`renderer.py`): zoompan com whip-zoom + shake matemático, 2.5D (fundo desfocado,
   sprite com balanço, sombra e entrada), poeira em blend screen, SFX nos cortes, trilha com ducking,
   legenda queimada.
7. **Publicação** (`src/publishers/`): YouTube (upload resumível, refresh de token, 1º comentário,
   flag de conteúdo sintético) e TikTok (Playwright com perfil persistente e proxy opcional).
8. **Relatório semanal** (`src/analytics_reporter.py`): tabela de 7 dias com diagnóstico de retenção,
   enviada ao Telegram e salva em `reports/` todo domingo às 20h.

---

## Setup

```bash
cp env.example .env            # preencha GEMINI_API_KEY e as senhas
docker compose up -d --build   # postgres, redis, bootstrap, workers e beat
```

Atenção: a senha dentro de `DATABASE_URL` precisa ser igual a `POSTGRES_PASSWORD`, com caracteres
especiais codificados na URL (`@` → `%40`, `#` → `%23`).

Autorização única por canal (no seu PC, com navegador — não dentro do Docker):

```bash
pip install -r requirements.txt && playwright install chromium
python main_cli.py --auth-youtube   # precisa de credentials/client_secret.json
python main_cli.py --auth-tiktok
```

## CLI

| Comando | O que faz |
|---|---|
| `python main_cli.py --init-assets` | Gera SFX/trilhas/overlay e cria a tabela `videos_history` |
| `python main_cli.py --auth-youtube [--force]` | OAuth do YouTube para canais sem token |
| `python main_cli.py --auth-tiktok [--force]` | Login do TikTok para perfis sem sessão |
| `python main_cli.py --init-mascots [--channel X] [--force]` | Gera a biblioteca de poses do mascote |
| `python main_cli.py --test-render <canal>` | Vídeo completo em `output_test/<canal>/`, sem publicar |
| `python main_cli.py --run-channel <canal> [--force]` | Pipeline completo, com publicação |
| `python main_cli.py --report-analytics` | Relatório de 7 dias no terminal + Telegram |

## Mascote fixo (canais 3D)

- Cada `*.png` em `media_assets/characters/<canal>/` é uma pose; o nome do arquivo é a chave
  que o roteiro usa. Nomes começando com `_` são ignorados.
- Pode trocar pelos seus PNGs: imagens sem transparência têm o fundo removido automaticamente
  (a cópia tratada fica em `_cache/`; o original nunca é alterado).
- `_preview.png` mostra a biblioteca inteira para revisão. Apague as poses que destoarem.

---

## Status (21/10/2026)

**Testado de verdade (Docker, chaves reais):** roteiro → narração → legenda → 16 imagens → render,
nos canais `nicho_1` (65,9 s) e `nicho_2` com mascote 2.5D (64,1 s); bibliotecas de mascote do
`nicho_2` (4 poses) e `nicho_4` (6 poses) geradas e revisadas.

**Escrito, ainda não testado:** uploads para YouTube e TikTok, relatório do YouTube Analytics,
B-roll do Pexels, alertas do Telegram, e um roteiro novo do Gemini usando as chaves de pose (o
Gemini ficou ~30 min respondendo 503 "high demand").

**Fora do escopo, de propósito:** técnicas anti-detecção de bot no TikTok (ocultar `webdriver`,
`AutomationControlled`, digitação "humanizada"). Fixar comentário não é possível pela API do YouTube.

## Pendências

1. Corrigir a `DATABASE_URL` no `.env` (ver Setup) — sem isso os workers do Celery não conectam.
2. Rodar `--auth-youtube` e `--auth-tiktok`.
3. Opcional: `POLLINATIONS_TOKEN` (vídeo cai de 13–22 min para ~5 min), `PEXELS_API_KEY`
   (B-roll), `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` (alertas).
4. Começar com `DRY_RUN=true` e revisar os vídeos antes de liberar a publicação.
5. Primeiro teste de publicação: `python main_cli.py --run-channel nicho_1`.

> Contexto completo (histórico, decisões, lições e próximos passos): [CONTEXT.md](CONTEXT.md)
