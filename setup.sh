#!/bash/bash
mkdir -p ~/src/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

echo "\
primaryColor="#bd93f9"\n\
backgroundColor="#282a36"\n\
secondaryBackgroundColor="#44475a"\n\
textColor="#f8f8f2"\n\
\n\
" > ./text.toml