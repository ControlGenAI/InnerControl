# InnerControl project page

Source of the project page for **InnerControl: Aligning ControlNet with Internal Feature Feedback** (WACV 2027).
The code lives on the `main` branch.

## Publishing

Settings → Pages → Build and deployment → Source: *Deploy from a branch*, Branch: `page`, folder `/ (root)`.

## Layout

```text
index.html          the page
static/css/         Bulma + page styles (index.css)
static/js/          Font Awesome icons
static/images/      figures from the paper, compressed for the web
```

Preview locally with `python3 -m http.server` in this folder and open http://localhost:8000.

The page is based on the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), adapted from [Nerfies](https://nerfies.github.io).
