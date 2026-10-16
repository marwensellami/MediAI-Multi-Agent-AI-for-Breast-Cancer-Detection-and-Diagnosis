import pydicom
ds = pydicom.dcmread("app/data/uploads/mammo.dcm")
print("Tags disponibles :")
for elem in ds:
    print(f"{elem.tag} : {elem.name} = {elem.value}")