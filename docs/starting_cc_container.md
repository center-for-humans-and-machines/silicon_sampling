# docker-compose.yml
```bash
docker compose -p silicon -f container/docker-compose.yml up -d cc_command_server --build && docker compose -p silicon -f container/docker-compose.yml exec cc_command_server bash -ic 'ssh -fN dais11 && sleep 0.5'
docker compose -p silicon -f container/docker-compose.yml run --build --rm claude-gpu
```