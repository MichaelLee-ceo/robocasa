# Sources of the README figures

`hierarchy.py` and `loop.py` write `hierarchy.html` and `loop.html` (inline SVG; fonts from Google Fonts). The frames `f<step>.jpg` are `video.robot0_agentview_right` of a replay of the
envkit policy test run SteamInMicrowave s2, at the end of each stage. Render with headless Chrome:
```bash
cd docs/ece780/src && python3 hierarchy.py && python3 loop.py
for f in hierarchy:1200,604 loop:1200,476; do n=${f%%:*}
  google-chrome --headless --no-sandbox --disable-gpu --hide-scrollbars --user-data-dir=/tmp/chrome-fig \
    --force-device-scale-factor=2 --window-size=${f#*:} --virtual-time-budget=6000 \
    --screenshot=$PWD/../$n.png "file://$PWD/$n.html"; done
rm -f hierarchy.html loop.html
```
