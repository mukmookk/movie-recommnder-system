#!/bash/bash
mkdir -p ~/.streamlit/

echo "> Executing setup.sh\n\n"

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

echo "[theme]\n\
primaryColor=\"#bd93f9\"\n\
backgroundColor=\"#282a36\"\n\
secondaryBackgroundColor=\"#44475a\"\n\
textColor=\"#f8f8f2\"\n\
\n\
" >> ~/.streamlit/config.toml

echo "> execution completed\n\n"

cat ~/.streamlit/config.toml