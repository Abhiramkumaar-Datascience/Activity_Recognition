document.getElementById('accelForm').addEventListener('submit', function(e){
    e.preventDefault();

    const xAccel = document.getElementById('x_accel').value;
    const yAccel = document.getElementById('y_accel').value;
    const zAccel = document.getElementById('z_accel').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            x_acceleration: xAccel,
            y_acceleration: yAccel,
            z_acceleration: zAccel
        }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerHTML = `<strong>Predicted Activity:</strong> ${data.predicted_activity}<br><strong>Description:</strong> ${data.Activity}`;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
