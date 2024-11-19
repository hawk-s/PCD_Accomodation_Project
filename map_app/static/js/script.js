document.addEventListener("DOMContentLoaded", function () {
    const map = L.map('map').setView([51.505, 10], 4); // Center of Europe
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
    }).addTo(map);

    const geojsonLayer = L.geoJSON().addTo(map);

    const updateMap = () => {
        const metric = document.getElementById('metric').value;
        const year = document.getElementById('year').value;
        const month = document.getElementById('month').value;

        fetch(`/data/${metric}/${year}/${month}`)
            .then(response => response.json())
            .then(data => {
                geojsonLayer.clearLayers();
                geojsonLayer.addData(data);
                geojsonLayer.setStyle({
                    color: 'blue',
                    weight: 1,
                    fillColor: 'cyan',
                    fillOpacity: 0.5
                });
                geojsonLayer.bindPopup(function (layer) {
                    return `<strong>Country:</strong> ${layer.feature.properties.country}<br>
                            <strong>Value:</strong> ${layer.feature.properties.value}`;
                });
            });
    };

    document.getElementById('updateMap').addEventListener('click', updateMap);
});
