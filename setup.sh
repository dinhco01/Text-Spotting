python setup.py build develop

wget -O ctw1500_attn_R_50.pth https://universityofadelaide.box.com/shared/static/okeo5pvul5v5rxqh4yg8pcf805tzj2no.pth

mkdir -p ~/.streamlit/

echo "\
[theme]
base='dark'\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml