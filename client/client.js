let endpointType = "detect-json";

const getApiUrl = () => {
  const url = `${window.location.protocol}//${window.location.hostname}:${window.location.port}/validator/${endpointType}`;
  console.log("[DEBUG] api url:", url);
  return url;
};

const mediaQuery = window.matchMedia("(max-width: 600px)");
console.log("mediaQuery", mediaQuery);

const logoImg = document.getElementById("logo-img");
logoImg.src = mediaQuery.matches
  ? "icons/metsnap-mobile-logo.png"
  : "icons/metsnap-web-logo.png";

let global_cache_image = null;
let global_cache_data = null;

const getVH = () => {
  return window.innerHeight;
};

const getLocalStorageSize = () => {
  let totalSize = 0;

  // Iterate over the keys in Local Storage
  for (let key in localStorage) {
    if (localStorage.hasOwnProperty(key)) {
      // Get the value associated with the key
      const value = localStorage.getItem(key);

      // Calculate the size of the key and value in bytes
      const keySize = new Blob([key]).size;
      const valueSize = new Blob([value]).size;

      // Add the size of the key and value to the total size
      totalSize += keySize + valueSize;
    }
  }

  return totalSize;
};

const clearResponse = () => {
  const canvasContainer = document.getElementById("canvas-container");
  const responseDiv = document.getElementById("json-content");
  canvasContainer.innerHTML = "";
  responseDiv.innerHTML = "";
};

const triggerFileSelect = () => {
  // Trigger the hidden file input when the button is clicked
  document.getElementById("fileInput").click();
};
// Optional: Listen for file selection and handle it
document
  .getElementById("fileInput")
  .addEventListener("change", function (event) {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      // Handle the selected file (e.g., upload it, display its name, etc.)
      console.log("Selected file:", selectedFile.name);

      // post_image(selectedFile);
      document.getElementById("burger-menu").classList.remove("close");
      document.getElementById("bur-menu-content").classList.remove("overlay");
      showImgInfoProcessing();
      rescaleAndPostImage(selectedFile);
    }
  });

const dmsToDecimal = (dmsArray) => {
  try {
    if (!dmsArray || dmsArray.length !== 3) {
      return null;
    }
    return dmsArray[0] + dmsArray[1] / 60 + dmsArray[2] / 3600;
  } catch {
    null;
  }
};

const rescaleAndPostImage = (file) => {
  let image = new Image();
  image.src = URL.createObjectURL(file);

  image.onload = () => {
    resizeAndSendImage(image, null);
  };

  image.onerror = function (e) {
    console.error("ERROR", e);
    showImgInfoError("Error occured while loading image!");
  };
};

const resizeAndSendImage = (image) => {
  let canvas = document.createElement("canvas");
  let ctx = canvas.getContext("2d");

  // Determine new dimensions
  let { width, height } = image;
  let scale = Math.min(720 / width, 720 / height);
  if (scale < 1) {
    width *= scale;
    height *= scale;
  }
  canvas.width = width;
  canvas.height = height;

  ctx.drawImage(image, 0, 0, width, height);
  canvas.toBlob((blob) => {
    post_image(blob);
  }, "image/jpeg");
};

const createFormData = (file) => {
  let formData = new FormData();
  formData.append("file", file);
  return formData;
};

const post_image = (file) => {
  const formData = createFormData(file);

  fetch(getApiUrl(), {
    method: "POST",
    body: formData,
  })
    .then((response) => handleFetchError(response))
    .then((data) => {
      console.log("DATA:", data);

      let image = new Image();
      image.src = URL.createObjectURL(file);
      image.onload = () => handleImageLoad(image, data);
      image.onerror = function (e) {
        console.error("ERROR", e);
        showImgInfoError("Error occured while loading image after analysis!");
      };
    })
    .catch((error) => {
      showImgInfoError(error.message);
      console.error(error);
    })
    .finally(() => {
      document.getElementById("fileInput").value = "";
    });
};

const computeDimensions = (image, parentContainer) => {
  // Get the aspect ratio of the image
  const aspectRatio = image.width / image.height;

  let computedWidth, computedHeight;

  if (mediaQuery.matches) {
    console.log("~~MOBILE");
    const parentWidth = window.innerWidth - 32;
    computedWidth = parentWidth;
    computedHeight = computedWidth / aspectRatio;
  } else {
    // Get the dimensions of the parent container
    const parentWidth = parentContainer.offsetWidth;
    const parentHeight = parentContainer.offsetHeight;
    console.log("~~DESKTOP");
    // Compute the dimensions based on the aspect ratio
    if (aspectRatio > 1) {
      // Horizontal image
      computedWidth = parentWidth;
      computedHeight = parentWidth / aspectRatio;
    } else if (aspectRatio < 1) {
      // Vertical image
      computedHeight = parentHeight;
      computedWidth = parentHeight * aspectRatio;
    } else {
      // Square image
      computedWidth = parentWidth;
      computedHeight = parentHeight;
    }
    // Ensure the computed dimensions do not exceed the parent container's dimensions
    computedWidth = Math.min(computedWidth, parentWidth);
    computedHeight = Math.min(computedHeight, parentHeight);
  }

  // Compute the scale factors
  const widthScaleFactor = computedWidth / image.width;
  const heightScaleFactor = computedHeight / image.height;

  return {
    width: computedWidth,
    height: computedHeight,
    widthScale: widthScaleFactor,
    heightScale: heightScaleFactor,
  };
};

const checkMasterCheckbox = () => {
  const checkboxes = Array.from(document.querySelectorAll(".cls-toggle input"));
  const checked = checkboxes.map((chBox) => chBox.checked);
  const masterCheckbox = document.getElementById("master-checkbox");
  masterCheckbox.checked = checked.every((value) => value === true);
};

const createMasterCheckbox = () => {
  const masterCheckbox = document.createElement("input");
  masterCheckbox.setAttribute("id", "master-checkbox");
  masterCheckbox.type = "checkbox";
  masterCheckbox.checked = true;

  masterCheckbox.addEventListener("change", function () {
    const checkboxes = document.querySelectorAll(".cls-toggle input");
    console.log("MASTER CHECKBOX", checkboxes);
    const checkTo = masterCheckbox.checked;
    checkboxes.forEach((checkbox) => {
      checkbox.checked = checkTo;
      checkbox.dispatchEvent(new Event("change"));
    });
  });

  const label = document.createElement("label");
  label.style.color = "white";
  label.appendChild(document.createTextNode("ALL"));

  const checkboxes = document.getElementById("checkboxes");
  const div = document.createElement("div");
  div.className = "master-cls-toggle";
  div.style.backgroundColor = "black";

  div.addEventListener("click", function (e) {
    if (e.target !== masterCheckbox) {
      masterCheckbox.checked = !masterCheckbox.checked;
      masterCheckbox.dispatchEvent(new Event("change")); // Trigger the change event for the checkbox
    }
  });

  div.appendChild(masterCheckbox);
  div.appendChild(label);
  checkboxes.appendChild(div);
};

const createCanvasForClass = (
  cls,
  color,
  backgroundColor,
  computedWidth,
  computedHeight
) => {
  const canvasContainer = document.getElementById("canvas-container");
  let canvas = document.createElement("canvas");

  canvas.width = computedWidth;
  canvas.height = computedHeight;
  canvas.className = "cls-canvas";
  canvas.id = `canvas-${cls}`;

  canvasContainer.appendChild(canvas);
  let ctx = canvas.getContext("2d");

  if (cls !== "main") {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.addEventListener("change", function () {
      canvas.style.display = checkbox.checked ? "block" : "none";
      checkMasterCheckbox();
    });

    const label = document.createElement("label");
    label.style.color = color;
    label.appendChild(document.createTextNode(cls));

    const checkboxes = document.getElementById("checkboxes");
    const div = document.createElement("div");
    div.className = "cls-toggle";
    div.style.backgroundColor = backgroundColor;

    div.addEventListener("click", function (e) {
      if (e.target !== checkbox) {
        checkbox.checked = !checkbox.checked;
        checkbox.dispatchEvent(new Event("change")); // Trigger the change event for the checkbox
      }
    });

    div.appendChild(checkbox);
    div.appendChild(label);

    checkboxes.appendChild(div);
  }

  return ctx;
};

const getColor = (uniqueClasses, cls) => {
  const _getColor = (cls, color_palette) => {
    let idx = uniqueClasses.indexOf(cls);
    idx %= color_palette.length;
    return color_palette[idx];
  };

  const color_palette = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "cyan",
    "brown",
    "grey",
    "pink",
    "purple",
  ];

  return _getColor(cls, color_palette);
};

const getBackgroundColor = (color) => {
  const color_palette = {
    red: "#FFDFD4",
    blue: "#7CB9E8",
    green: "#ACE1AF",
    yellow: "#C5A139",
    orange: "#9A8271",
    cyan: "#79A9A8",
    brown: "#DDAA9A",
    grey: "#EAEAEB",
    pink: "#987883",
    purple: "#E6DBEE",
  };
  return color_palette[color];
};

const drawBboxes = (image, boxes, classes, uniqueClasses, dimensions) => {
  const scaleFactor = Math.sqrt(dimensions.width * dimensions.height) / 500;
  const classCanvases = {};

  createMasterCheckbox();

  const cenzured = [];

  for (let i = 0; i < boxes.length; i++) {
    const bbox = boxes[i].map((coord, index) => {
      // Scale x-coordinates by widthScale and y-coordinates by heightScale
      return (
        coord *
        (index % 2 === 0 ? dimensions.widthScale : dimensions.heightScale)
      );
    });
    console.log("RESCALED CENZURED");
    const text = classes[i].replace(" ", "_");
    if (
      text.startsWith("EXPOSED") ||
      text == "face" ||
      text == "license_plate"
    ) {
      cenzured.push(bbox);
    }
  }

  if (cenzured.length) {
    cenzuredCtx = createCanvasForClass(
      "cenzured",
      "red",
      "yellow",
      dimensions.width,
      dimensions.height
    );
  }

  for (let i = 0; i < boxes.length; i++) {
    const bbox = boxes[i].map((coord, index) => {
      // Scale x-coordinates by widthScale and y-coordinates by heightScale
      return (
        coord *
        (index % 2 === 0 ? dimensions.widthScale : dimensions.heightScale)
      );
    });

    const text = classes[i].replace(" ", "_");
    const color = getColor(uniqueClasses, classes[i]);
    const backgroundColor = getBackgroundColor(color);

    let width = bbox[2] - bbox[0]; // Calculate width
    let height = bbox[3] - bbox[1]; // Calculate height

    // If there's no canvas for this class yet, create one
    if (!classCanvases[text]) {
      classCanvases[text] = createCanvasForClass(
        text,
        color,
        backgroundColor,
        dimensions.width,
        dimensions.height
      );
    }

    let ctx = classCanvases[text]; // Get the canvas context for this class

    // Draw a filled rectangle with opacity
    ctx.globalAlpha = 0.3; // Adjust transparency here
    ctx.fillStyle = color;
    ctx.fillRect(bbox[0], bbox[1], width, height);

    // Reset globalAlpha and draw the box outline
    ctx.globalAlpha = 1.0;
    ctx.beginPath();
    ctx.rect(bbox[0], bbox[1], width, height);
    ctx.lineWidth = Math.max(1, scaleFactor * 3);
    ctx.strokeStyle = color;
    ctx.stroke();

    let textPosition = Math.min(
      Math.max(bbox[1], scaleFactor * 20),
      dimensions.height - (Math.max(10, scaleFactor * 14) + 5)
    );

    ctx.font = `${Math.max(10, scaleFactor * 14)}px Arial`;
    ctx.fillStyle = color;
    ctx.fillText(`${text}: ${boxes[i][4].toFixed(2)}`, bbox[0], textPosition);

    console.log(ctx);
  }

  // cenzure  // cenzure  // cenzure //
  // cenzure  // cenzure  // cenzure //
  // c // e // n // z // u // r // e //

  if (cenzured.length) {
    cenzuredCtx.drawImage(image, 0, 0, dimensions.width, dimensions.height);
    // cenzuredCtx.filter = "blur(50px)"; // Set the blur radius to 5 pixels
    // cenzuredCtx.drawImage(cenzuredCtx.canvas, 0, 0); // Draw the image back onto itself
    // cenzuredCtx.filter = "none"; // Reset the filter to none
  }

  for (const cenzuredBbox of cenzured) {
    const [x_min, y_min, x_max, y_max, conf, cls] = cenzuredBbox;
    const width = x_max - x_min;
    const height = y_max - y_min;

    const imgData = cenzuredCtx.getImageData(x_min, y_min, width, height);

    const offscreenCanvas = document.createElement("canvas");
    const offscreenCtx = offscreenCanvas.getContext("2d");

    offscreenCanvas.width = width;
    offscreenCanvas.height = height;
    offscreenCtx.putImageData(imgData, 0, 0);

    const pixelation = 24;

    const scaledCanvas = document.createElement("canvas");
    const scaledCtx = scaledCanvas.getContext("2d");
    scaledCanvas.width = Math.ceil(width / pixelation);
    scaledCanvas.height = Math.ceil(height / pixelation);

    scaledCtx.drawImage(
      offscreenCanvas,
      0,
      0,
      width,
      height,
      0,
      0,
      Math.ceil(width / pixelation),
      Math.ceil(height / pixelation)
    );

    cenzuredCtx.drawImage(
      scaledCanvas,
      0,
      0,
      Math.ceil(width / pixelation),
      Math.ceil(height / pixelation),
      x_min,
      y_min,
      width,
      height
    );
  }
};

//
//
// Screens
//
//

const showImgInfoWelcome = () => {
  document.getElementById("fileInput").value = "";
  document.getElementById("img-content").style.display = "flex";
  document.getElementById("img-content").style.width = mediaQuery.matches
    ? "100vw"
    : "calc(100vw - 150px)";

  mediaQuery.matches
    ? (document.getElementById(
        "img-content"
      ).style.height = `calc(${getVH()}px - 80px)`)
    : (document.getElementById("img-content").style.height = "100vh");

  document.getElementById("json-content").style.display = "none";

  document.getElementById("img-info-div").style.display = "flex";
  document.getElementById("img-content-div").style.display = "none";
  document.getElementById("img-info-welcome").style.display = "flex";
  document.getElementById("img-info-error").style.display = "none";
  document.getElementById("img-info-processing").style.display = "none";
};

const showImgInfoProcessing = () => {
  clearResponse();
  document.getElementById("img-content").style.display = "flex";
  document.getElementById("img-content").style.width = mediaQuery.matches
    ? "100vw"
    : "calc(100vw - 150px)";

  mediaQuery.matches
    ? (document.getElementById(
        "img-content"
      ).style.height = `calc(${getVH()}px - 80px)`)
    : (document.getElementById("img-content").style.height = "100vh");

  document.getElementById("json-content").style.display = "none";

  document.getElementById("img-info-div").style.display = "flex";
  document.getElementById("img-content-div").style.display = "none";

  document.getElementById("img-info-welcome").style.display = "none";
  document.getElementById("img-info-error").style.display = "none";
  document.getElementById("img-info-processing").style.display = "flex";
};

const showImgInfoError = (error) => {
  document.getElementById("img-info-div").style.display = "flex";
  document.getElementById("img-content-div").style.display = "none";
  document.getElementById("img-info-welcome").style.display = "none";
  document.getElementById("img-info-error").style.display = "flex";
  document.getElementById("img-info-processing").style.display = "none";

  document.getElementById("img-info-error-message").innerHTML = error;
};

const showImgContent = () => {
  document.getElementById("img-content").style.display = "flex";
  document.getElementById("img-content").style.width = mediaQuery.matches
    ? "100vw"
    : "calc(65vw - 150px)";

  mediaQuery.matches
    ? (document.getElementById("img-content").style.height = `100%`)
    : (document.getElementById("img-content").style.height = "100vh");

  document.getElementById("json-content").style.display = "flex";

  document.getElementById("img-info-div").style.display = "none";
  document.getElementById("img-content-div").style.display = "flex";
};

//
//
// Component generators
//
//

const generateODValidators = (data) => {
  const odValidators = {
    vehicle: {
      name: "Vehicle detection validation",
      infoPassed:
        "No license plates or SIGNIFICANT vehicles detected in the image",
      infoFailed:
        "Some license plates or SIGNIFICANT vehicles detected in the image",
    },
    person: {
      name: "Person detection validation",
      infoPassed: "No faces or SIGNIFICANT people detected in the image",
      infoFailed: "Some faces or SIGNIFICANT people detected in the image",
    },
    nudity: {
      name: "Nudity detection validation",
      infoPassed: "No nudity detected in the image",
      infoFailed: "Some nudity detected in the image",
    },
    sky: {
      name: "Sky detection validation",
      infoPassed: "Sky that meets all requirements detected in the image",
      infoFailed: "No sky detected in the image that meets all requirements",
    },
    text: {
      name: "Text detection validation",
      infoPassed: "No SIGNIFICANT texts detected in the image",
      infoFailed: "Some SIGNIFICANT texts detected in the image",
    },
  };

  const responseDiv = document.getElementById("json-content");
  const od_val_div = document.createElement("div");
  responseDiv.appendChild(od_val_div);

  od_val_div.setAttribute("class", "validators-div");

  for (const [key, value] of Object.entries(data)) {
    if (!odValidators[key]) {
      continue;
    }

    console.log("key:", key, value);

    const status = value ? "Passed" : "Failed";
    const info = value
      ? odValidators[key].infoPassed
      : odValidators[key].infoFailed;

    console.log("status:", status);
    console.log("info:", info);

    component = createValidatorComponent(odValidators[key].name, status, info);
    od_val_div.appendChild(component);
  }
};

const generateIsValid = (data) => {
  const status = data.is_valid ? "VALID" : "NOT VALID";
  const statusSvg = data.is_valid ? "Passed" : "Failed";
  const info = data.is_valid
    ? "Image is valid because it passed all validations"
    : "Image is not valid beacause it did not pass all validations";
  const responseDiv = document.getElementById("json-content");

  const is_val_div = document.createElement("div");
  responseDiv.appendChild(is_val_div);

  is_val_div.setAttribute("class", "validators-div");

  component = createValidatorComponent("STATUS", status, info, statusSvg);
  component.querySelector(".validator-prop-status").id =
    "status-validator-prop-status";

  console.log("VALID COMP:", component);
  if (data.is_valid) {
    component.setAttribute(
      "style",
      "background-color: #b7d5ad; border-radius: 12px"
    );
  } else {
    component.setAttribute(
      "style",
      "background-color: #f0BEBD; border-radius: 12px"
    );
  }
  is_val_div.appendChild(component);
};

const generateWeatherComponent = (data) => {
  const weather = data.weather[0];

  const responseDiv = document.getElementById("json-content");
  const validatorsDiv = document.createElement("div");
  responseDiv.appendChild(validatorsDiv);

  validatorsDiv.className = "validators-div";

  // Create the category container
  const weatherDiv = document.createElement("div");
  weatherDiv.className = "category-div";

  // Create the category label
  const weatherLabel = document.createElement("div");
  weatherLabel.className = "weather-label";
  const strongLabel = document.createElement("strong");
  strongLabel.textContent = "Classified weather";
  weatherLabel.appendChild(strongLabel);

  // Create the categories list
  const weatherCategoriesDiv = document.createElement("div");
  weatherCategoriesDiv.id = "weather-categories";

  const categoryDiv = document.createElement("div");
  categoryDiv.setAttribute("class", "validator-prop-status-cat");
  const categoryImg = document.createElement("img");
  categoryImg.setAttribute("src", `icons/${weather}.png`);
  categoryImg.setAttribute("width", "40px");
  categoryImg.setAttribute("alt", "Status");
  categoryImg.style.marginRight = "1em";
  const categoryText = document.createElement("div");
  categoryText.setAttribute("class", "status-text");
  categoryText.textContent = weather;
  categoryDiv.appendChild(categoryImg);
  categoryDiv.appendChild(categoryText);
  weatherCategoriesDiv.appendChild(categoryDiv);

  // Append everything together
  weatherDiv.appendChild(weatherLabel);
  weatherDiv.appendChild(weatherCategoriesDiv);
  validatorsDiv.appendChild(weatherDiv);

  // Return the main container
  return validatorsDiv;
};

const generateCategoryComponent = (data) => {
  const categories = data.category;

  const responseDiv = document.getElementById("json-content");
  const validatorsDiv = document.createElement("div");
  responseDiv.appendChild(validatorsDiv);

  validatorsDiv.className = "validators-div";

  // Create the category container
  const categoryDiv = document.createElement("div");
  categoryDiv.className = "category-div";

  // Create the category label
  const categoryLabel = document.createElement("div");
  categoryLabel.className = "category-label";
  const strongLabel = document.createElement("strong");
  strongLabel.textContent = "Classified categories";
  categoryLabel.appendChild(strongLabel);

  // Create the categories list
  const categoriesDiv = document.createElement("div");
  categoriesDiv.id = "categories";

  if (categories.length < 1) {
    categories.push("unknown");
  }

  console.log("CATEGORIES:", categories);

  // Loop through the categories array and create each list item
  categories.forEach((category) => {
    const categoryDiv = document.createElement("div");
    categoryDiv.setAttribute("class", "validator-prop-status-cat");
    const categoryImg = document.createElement("img");
    categoryImg.setAttribute("src", `icons/${category}.png`);
    categoryImg.setAttribute("width", "40px");
    categoryImg.setAttribute("alt", "Status");
    categoryImg.style.marginRight = "1em";
    const categoryText = document.createElement("div");

    if (category === "unknown") {
      categoryText.textContent = "no category classified";
    } else {
      categoryText.setAttribute("class", "status-text");
      categoryText.textContent = category;
    }

    categoryDiv.appendChild(categoryImg);
    categoryDiv.appendChild(categoryText);
    categoriesDiv.appendChild(categoryDiv);
  });

  // Append everything together
  categoryDiv.appendChild(categoryLabel);
  categoryDiv.appendChild(categoriesDiv);
  validatorsDiv.appendChild(categoryDiv);

  // Return the main container
  return validatorsDiv;
};

const generateCVValidators = (data) => {
  cvValidators = {
    vertical_corruption: {
      name: "Vertical Distortion Check",
      infoPassed: "No vertical distortion detected in the image",
      infoFailed: "Vertical distortion detected in the image",
    },
    partial_download: {
      name: "Download Status Check",
      infoPassed: "Image has been completely downloaded",
      infoFailed: "Image has been downloaded partially",
    },
    no_image: {
      name: "Content Existence Check",
      infoPassed: "Content detected in the image",
      infoFailed: "No content detected in the image",
    },
  };

  const responseDiv = document.getElementById("json-content");
  const cv_val_div = document.createElement("div");
  responseDiv.appendChild(cv_val_div);

  cv_val_div.setAttribute("class", "validators-div");

  for (const [key, value] of Object.entries(data)) {
    if (!cvValidators[key]) {
      continue;
    }

    const status = value ? "Passed" : "Failed";
    const info = value
      ? cvValidators[key].infoPassed
      : cvValidators[key].infoFailed;

    component = createValidatorComponent(cvValidators[key].name, status, info);
    cv_val_div.appendChild(component);
  }
};

const createValidatorComponent = (validatorName, status, info, svg_name) => {
  svg_name = svg_name || status;
  // Create main div
  const mainDiv = document.createElement("div");
  mainDiv.setAttribute("class", "validator-prop-div");

  // Create validator name div
  const nameDiv = document.createElement("div");
  nameDiv.setAttribute("class", "validator-prop-name");
  const nameStrong = document.createElement("strong");
  nameStrong.textContent = validatorName;
  nameDiv.appendChild(nameStrong);

  // Create status div
  const statusDiv = document.createElement("div");
  statusDiv.setAttribute("class", "validator-prop-status");
  const statusImg = document.createElement("img");
  statusImg.setAttribute("src", `icons/${svg_name}.svg`);
  statusImg.setAttribute("width", "24px");
  statusImg.setAttribute("alt", "Status");
  statusImg.style.marginRight = "1em";
  const statusText = document.createElement("div");
  statusText.setAttribute("class", "whole-status-text");
  statusText.textContent = status;
  statusDiv.appendChild(statusImg);
  statusDiv.appendChild(statusText);
  nameDiv.appendChild(statusDiv);

  // Create info div
  const infoDiv = document.createElement("div");
  infoDiv.setAttribute("class", "validator-prop-info");
  infoDiv.textContent = info;

  // Append all to main div
  mainDiv.appendChild(nameDiv);
  mainDiv.appendChild(infoDiv);

  return mainDiv;
};

const generateImg = (image, data) => {
  const canvasContainer = document.getElementById("canvas-container");
  canvasContainer.innerHTML = "";
  document.getElementById("checkboxes").innerHTML = "";
  const dimensions = computeDimensions(image, canvasContainer);

  console.log("dimensions:", dimensions);

  if (mediaQuery.matches) {
    document.getElementById(
      "img-content"
    ).style.height = `calc(${dimensions.height}px + 60px + 2em)`;
    document.getElementById(
      "img-content-div"
    ).style.height = `calc(${dimensions.height}px + 60px + 2em)`;
    document.getElementById(
      "img-div"
    ).style.height = `calc(${dimensions.height}px + 2em)`;
  }

  let imgCtx = createCanvasForClass(
    "main",
    "black",
    "black",
    dimensions.width,
    dimensions.height
  );

  imgCtx.drawImage(image, 0, 0, dimensions.width, dimensions.height);

  const uniqueClasses = [...new Set(data.classes)];

  console.log(data);
  if (data.boxes) {
    console.log("data boxes");
    drawBboxes(image, data.boxes, data.classes, uniqueClasses, dimensions);
  }
};

//
//
//HANDLERS
//
//

const handleFetchError = (response) => {
  if (!response.ok) {
    const errorMessage = `Error: response status is ${response.status} ${response.statusText}`;
    showImgInfoError(errorMessage);
    throw new Error(errorMessage);
  }
  return response.json();
};

const handleImageLoad = (image, data, coords) => {
  showImgContent();
  clearResponse();

  // cache data
  global_cache_image = image;
  global_cache_data = data;

  generateImg(image, data);
  data.hasOwnProperty("is_valid") && generateIsValid(data);

  if (data.hasOwnProperty("weather")) {
    generateWeatherComponent(data);
  }

  if (data.hasOwnProperty("category")) {
    generateCategoryComponent(data);
  }
  generateODValidators(data);
  generateCVValidators(data);
};

// BURGER MENU

const burgerMenu = document.getElementById("burger-menu");
const overlay = document.getElementById("bur-menu-content");
burgerMenu.addEventListener("click", function () {
  this.classList.toggle("close");
  overlay.classList.toggle("overlay");
});

// BURGER MENU

// -----
// START

showImgInfoWelcome();

window.addEventListener("resize", function () {
  if (!!global_cache_image && !!global_cache_data) {
    generateImg(global_cache_image, global_cache_data);
  }
});

// START
// -----
