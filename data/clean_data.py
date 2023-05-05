import json

filename = "instruction.json"
file = open(filename)
instructions = json.load(file)
cleaned_instructions = []
blacklist = [
    "HTML",
    "SQL",
    "C++",
    " C ",
    "Ruby",
    "Java",
    "JavaScript",
    "Swift",
    "Visual Basic",
    "Bash script",
    "Objective-C",
    "CSS",
    "C#",
    "console.log",
    "Ruby"
]

for instruction in instructions:

    skip = False
    for keyword in blacklist:
        if keyword in instruction["instruction"]:
            skip = True
        if keyword.lower() in instruction["instruction"]:
            skip = True

    if skip:
        continue
    else:
        cleaned_instructions.append(instruction)

py_instructions = []

for instruction in instructions:

    skip = True
    if "Python" in instruction["instruction"]:
        skip = False
    if "def" in instruction["output"]:
        skip = False

    if skip:
        continue
    else:
        py_instructions.append(instruction)

filename = "instruction_alpaca_python.json"
file = open(filename, encoding="cp866")
instructions = json.load(file)

for instruction in instructions:

    skip = True
    if "Python" in instruction["instruction"]:
        skip = False

    if "def" in instruction["output"]:
        skip = False

    if skip:
        continue
    else:
        py_instructions.append(instruction)

# Dump to Json again
with open("pyinstructions.json", "w") as final:
    json.dump(py_instructions, final)
