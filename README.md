# Using a feature engineered dataset from a previous assignment, answer these three questions:

## Question 1
What price could we estimate for a car with the following features:

answers = {}

#MISSING: PRECIO, MODEL, VERSION1, "N° marchas"
question1 = {
    "BRAND":"Toyota",
    "POTENCIA1 (cv)":280,
    "PUERTAS":4.0,
    "COMBUSTIBLE":"Gasolina",
    "CO2":"ambar",
    "Tracción":"Tracción total permanente ",
    "Transmisión":"Automática secuencial ",
    "Emisiones CO2 (gr/km)":154.0,
    "Autonomía (km)":840.6,
    "Consumo (l/100km)":6.9,
    "Garantía (meses)":24.0,
    "Motor (cc)":1995.0,
    "Capacidad depósito (lts)":58.0,
    "Velocidad Máxima (km/h)":240.0,
    "Aceleración (0-100 km) (s)":5.2,
    "Revoluciones Potencia Máxima (rpm)":5250.0,
    "Par motor (Nm)":400.0,
    "Carrocería":"Berlina ",
    "Peso (kg)":1530.0,
    "Largo (mm)":4650.0,
    "Ancho (mm)":1860.0,
    "Alto (mm)":1438.0,
}


## Question 2
What would be the expected range of prices for a a car with the following features:
* BRAND: Mercedes
* PUERTAS: 2 (two doors)
* POTENCIA1 (cv) between 200 and 300 (200 hp < power < 300 hp)



## Question 3
If a given car weights 1500 kg (```Peso (kg)```) and is 4600 mm long (```Largo (mm)```), 1900 mm wide (```Ancho (mm)```) and 1400 mm tall (```Alto (mm)```), what could be its body type (```Carrocería```) (assign probabilities to your opinion)


question3 = {
    "Peso (kg)":1500.0,
    "Largo (mm)":4600.0,
    "Ancho (mm)":1900.0,
    "Alto (mm)":1400.0
}



Done in a group assignment with Isa, Juan Diego
