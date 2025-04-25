# 切换到项目的 serve 目录，确保后续命令在正确路径下执行
! cd serve

# 使用 uvicorn 启动 FastAPI 服务，--reload 表示代码变动时自动重启
! uvicorn api:app --reload