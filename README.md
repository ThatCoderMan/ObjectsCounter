| Train Model                                                                                                                                                                       | Detect Images                                                                                                                                                                          |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![train model](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThatCoderMan/ObjectsCounter/blob/main/notebooks/yolov8.ipynb) | [![detect](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThatCoderMan/ObjectsCounter/blob/main/notebooks/HaystackDetector.ipynb) |

# Objects Detection

<details>
<summary>Project stack</summary>

 - Python 3.11
 - Ultralytics
 - Roboflow
 - opencv
 - argparse

</details>

---

## Description

Object detecting, tracking and counting with YOLO 8 model.
<h4>
<details>
<summary>Cars tracker example </summary>

![track cars.gif](docs%2Ftrack%20car.gif)
</details>

<details>
<summary>Hay counter example</summary>

![hay counter.gif](docs%2Fhay%20counter.gif)
</details>


## Launch Instructions

Clone repository:

```bash
git clone git@github.com:ThatCoderMan/ObjectsCounter.git
```

Install and activate the virtual environment:

<details>
<summary>MacOS</summary>

```bash
source venv/bin/activate
```
</details>
<details>
<summary>Windows</summary>

```bash
python -m venv venv
source venv/Scripts/activate
```
</details>

Install the dependencies from the file requirements.txt:

```bash
pip install -r requirements.txt
```

Run the command to run:

```bash
python main.py
```

---

## Developers:

- [ThatCoderMan (Артемий)](https://github.com/ThatCoderMan)
- [progmat64 (Александр)](https://github.com/progmat64)
- [Bully-Boy (Роман)](https://github.com/Bully-Boy)