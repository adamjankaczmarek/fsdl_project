import torch
import logging
from ts.torch_handler.base_handler import BaseHandler
from wav2keyword import KWS
from scipy.io.wavfile import read
import io

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        w2v_file = self.manifest['model']['serializedFile']
        w2k_file = self.manifest['model']['modelFile']
        
        w2v_sd = torch.load(w2v_file)
        w2k_sd = torch.load(w2k_file)
        encoder_prefix = "model.w2v_encoder."
        encoder_prefix_len = len(encoder_prefix)
        w2v_encoder_weights = {
            weight_name[encoder_prefix_len:]:weight 
            for weight_name, weight in w2k_sd['state_dict'].items()
            if weight_name.startswith(encoder_prefix)
        }
        decoder_prefix = "model.decoder."
        decoder_prefix_len = len(decoder_prefix)
        w2v_decoder_weights = {
            weight_name[decoder_prefix_len:]:weight
            for weight_name, weight in w2k_sd['state_dict'].items()
            if weight_name.startswith(decoder_prefix)
        }

        self.model = KWS(22, 768, w2v_sd)
        self.model.w2v_encoder.load_state_dict(w2v_encoder_weights)
        self.model.decoder.load_state_dict(w2v_decoder_weights)
        self.model.to(self.device)
        self.model.eval()
        #serialized_file = self.manifest['model']['serializedFile']
        #model_pt_path = os.path.join(model_dir, serialized_file)
        #if not os.path.isfile(model_pt_path):
        #    raise RuntimeError("Missing the model.pt file")

        #self.model = torch.jit.load(model_pt_path)

        self.initialized = True


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        logger.warning(data)
        rate, data = read(io.BytesIO(data[0]['body']))
        sample_rate = 16000
        if data.shape[0] < sample_rate:
            pad_size = sample_rate - data.shape[0]
            data = np.pad(data, (round(pad_size/2)+1, round(pad_size/2)+1), 'constant', constant_values=0)
        mid = int(data.shape[0]/2)
        cut_off = int(sample_rate/2)
        data = data[mid-cut_off:mid+cut_off]
        logger.warning(data)
        logger.warning(data.shape)
        logger.warning(torch.tensor([data]).shape)
        pred_out = self.model.forward({"source": torch.FloatTensor([data]).to(self.device)})
        return [pred_out.cpu().detach().numpy().tolist()]

