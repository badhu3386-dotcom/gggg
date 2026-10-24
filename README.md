# Sign to Voice

A website that reads sign language through your laptop webcam, shows the result as text on screen, and speaks it aloud.

- **Hand tracking:** [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) finds 21 landmark points on your hand.
- **Recognition:** a small k-Nearest-Neighbours classifier compares your hand pose against signs you have taught it.
- **Voice:** the browser's built-in Web Speech API (English, Hindi, Malayalam, Tamil, and more, depending on the voices installed on your device).
- **Privacy:** everything runs in the browser. No video leaves your computer.

## Get your own signs file quickly (Guided setup)

Use the **Guided setup** card: choose a pack (common words, numbers, or letters), then for each sign shown, make the correct ISL sign, press **Record** and hold it steady. When done, click **Export signs** to download `my-signs.json`. You can re-import that file any time (or on another device) with **Import signs**.

## How to use

1. Click **Start camera** and allow access.
2. Under **Teach a sign**, type a word (e.g. `Hello`), make the sign, press **Record**. Repeat 2-3 times with small variations. Add as many signs as you like.
3. Make a sign and hold it steadily. The word appears in **Translation** and is spoken.
4. Use **Export signs** to back up your data, and **Import signs** to load it on another device.

Start with 5-10 signs. Once those work well, add more.

## Run locally

The camera only works on `https` or `localhost`, so don't just double-click `index.html`. From this folder:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000> in Chrome or Edge.

## Put it online with GitHub Pages

1. Create a new GitHub repository and upload all files from this folder.
2. Go to **Settings -> Pages**, set the source to the `main` branch and root folder, and save.
3. After a minute your site is live at `https://<your-username>.github.io/<repo-name>/` (https, so the camera works).

## Project files

| File | Purpose |
|------|---------|
| `index.html` | Page layout |
| `style.css` | Styling |
| `app.js` | Camera, hand tracking, classifier, speech |

## Tips for better accuracy

- Good, even lighting and a plain background.
- Record each sign several times, at slightly different distances and angles.
- Make signs look clearly different from each other.
- Adjust **Hold time** and **Strictness** sliders in the Translation panel.

## Limitations and next steps

- Tracks **one or two hands** and recognises **static poses** (letters, numbers, simple words). Signs that depend on movement need more work.
- Ideas: motion signs using sequences of frames (LSTM), a pre-trained ISL dataset, sentence building.

## Optional: build the signs file from a dataset (Python)

`tools/make_signs_file.py` reads a folder of ISL videos/images, runs MediaPipe on them, and writes a `my-signs.json` you can load with **Import signs**.

```bash
cd tools
pip install -r requirements.txt
python make_signs_file.py --dataset "PATH/TO/DATASET" --words "hello,thank you,love you,help,welcome" --out my-signs.json
```

The script prints which words it found and which were missing. Only words that exist in your dataset folder (as folder or file names) can be built. Check the dataset's licence before using or sharing it.
