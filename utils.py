class BoxPointsConverter:
    
    @staticmethod
    def voc_to_yolo(box):
        # Convert [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height]
        class_id, x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        return [class_id, x_center, y_center, width, height]
    
    @staticmethod
    def voc_to_coco(box):
        # Convert [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
        class_id, x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        return [class_id, x_min, y_min, width, height]

    @staticmethod
    def yolo_to_voc(box):
        # Convert [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]
        class_id, x_center, y_center, width, height = box
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        return [class_id, x_min, y_min, x_max, y_max]
    
    @staticmethod
    def yolo_to_coco(box):
        # Convert [x_center, y_center, width, height] to [x_min, y_min, width, height]
        class_id, x_center, y_center, width, height = box
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        return [class_id, x_min, y_min, width, height]
    
    @staticmethod
    def coco_to_yolo(box):
        # Convert [x_min, y_min, width, height] to [x_center, y_center, width, height]
        class_id, x_min, y_min, width, height = box
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        return [class_id, x_center, y_center, width, height]
    
    @staticmethod
    def coco_to_voc(box):
        # Convert [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
        class_id, x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        return [class_id, x_min, y_min, x_max, y_max]