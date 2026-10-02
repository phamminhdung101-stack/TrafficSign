function chooseImage() {

    const input = document.getElementById("imageInput");

    input.click();

    input.onchange = function () {

        const file = input.files[0];

        if (file) {

            const image = document.getElementById("previewImage");
            const text = document.getElementById("previewText");

            image.src = URL.createObjectURL(file);

            image.style.display = "block";

            text.style.display = "none";
        }
    };
}
async function openCamera() {

    const camera = document.getElementById("camera");
    const text = document.getElementById("previewText");

    try {

        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        camera.srcObject = stream;

        camera.style.display = "block";

        text.style.display = "none";

    } catch (error) {

        alert("Không thể mở camera. Vui lòng cấp quyền sử dụng camera.");

        console.error(error);
    }
}
function chooseVideo() {

    const input = document.getElementById("videoInput");

    input.click();

    input.onchange = function () {

        const file = input.files[0];

        if (file) {

            const video = document.getElementById("camera");
            const image = document.getElementById("previewImage");
            const text = document.getElementById("previewText");

            video.srcObject = null;
            video.src = URL.createObjectURL(file);

            video.controls = true;
            video.style.display = "block";

            image.style.display = "none";
            text.style.display = "none";

            video.play();
        }
    };
}
function recognizeImage() {

    const resultStatus = document.getElementById("resultStatus");
    const resultInfo = document.getElementById("resultInfo");

    resultStatus.innerHTML = "⏳ Đang thực hiện nhận diện...";

    resultInfo.style.display = "none";

    setTimeout(function() {

        resultStatus.innerHTML = "✅ Nhận diện hoàn tất!";

        document.getElementById("signName").innerHTML =
            "Biển báo cấm đi ngược chiều";

        document.getElementById("confidence").innerHTML =
            "95%";

        resultInfo.style.display = "block";

    }, 1000);
}