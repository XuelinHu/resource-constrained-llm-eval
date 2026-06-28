const path = require('path')

const root = __dirname

module.exports = {
  apps: [
    {
      name: 'railway-corpus-api',
      cwd: path.join(root, 'backend'),
      script: '/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python',
      args: '-m uvicorn app.main:app --host 0.0.0.0 --port 8000',
      interpreter: 'none',
      autorestart: true,
      max_restarts: 10,
      time: true,
    },
    {
      name: 'railway-corpus-frontend',
      cwd: path.join(root, 'frontend'),
      script: 'npm',
      args: 'run dev:frp -- --host 0.0.0.0 --port 4005',
      interpreter: 'none',
      autorestart: true,
      max_restarts: 10,
      time: true,
    },
  ],
}
