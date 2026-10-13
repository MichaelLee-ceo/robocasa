import asyncio, sys
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium" if __import__("os").path.exists("/opt/pw-browsers/chromium") else None)
        for w, name in ((1280, "desk"), (390, "mobile")):
            pg = await b.new_page(viewport={"width": w, "height": 900}); errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            await pg.goto("file://" + sys.argv[1]); await pg.wait_for_timeout(800)
            el = await pg.query_selector("#extra")
            print(name, "errors", errs, "extra cards", len(await pg.query_selector_all("#extra .card")), "scrollW", await pg.evaluate("document.documentElement.scrollWidth"))
            await el.screenshot(path=f"/tmp/claude-1001/-home-michaellee-Desktop-robocasa/d2b6537f-619c-5af7-ab37-a54e14a4b87a/scratchpad/extra_{name}.png")
        await b.close()
asyncio.run(main())
