## Trabajo practico UTN FRBA machine learning

Archivo data.csv en el aula virtual

### Idea general

- Procesar el data.csv para generar una ABT con un registro unico por cliente
- Filtros a aplicar:
  - Clientes con historial completo (un registro cada mes del universo original)
  - Clientes sin paquete activo en el ultimo mes del training window
  - Clientes sin cobranding en el ultimo mes del training window
  - Limpiar inconsistencias o filas con errores
  - Eliminar duplicados, tiene que cada client_id tener un solo registro por mes y/o periodo
- Identificar identiry features
- Identificar transform features
- Tratar variables con nulos
- Limpiar la data de outliers

### Posibles identity o transform features

- Cantidad de seguros contratados
- Cantidad de productos activos
- Transacciones segun el medio (mobile, web, etc) y % de tipo de transaccion sobre total de transacciones
- Agregar mean, median, percentiles, etc de todo lo numerico entre meses
- Indexar por CER los valores monetarios
- Sacar outlaierts de todo lo numerico utilizando revisando la distribución
- Completar nulos segun el caso (media probablemente o valor no nulo previo)
