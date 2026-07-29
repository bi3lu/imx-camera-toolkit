if isinstance(value, str):
  result.append(value)
else:
  for subkey, subvalue in value.items():
    if subkey == "Valid for Units:":
      name = {key: value}
    elif isinstance(subvalue, dict) and len(subvalue) == 1:
      name = {
        key: {list(subvalue.keys())[0]: list(subvalue.values())[0]}
      }
    else:
      name = {key: subvalue}
    result.append(name)  # This is where the error occurs 