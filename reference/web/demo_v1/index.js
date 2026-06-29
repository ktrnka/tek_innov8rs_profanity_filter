// npm i @huggingface/transformers
import { pipeline } from '@huggingface/transformers';

// Allocate pipeline
async function loadModel() {
    const pipe = await pipeline('text-classification', 'keith-trnka/hgjghjghj');
    return pipe;
}

// Initialize the model
loadModel().then(pipe => {
    console.log('Model loaded successfully');
    // You can use pipe here or store it globally
    window.profanityFilter = pipe;
}).catch(err => {
    console.error('Failed to load model:', err);
}); 