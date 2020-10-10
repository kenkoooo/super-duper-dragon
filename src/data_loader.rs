use tch::kind::Element;
use tch::Tensor;

pub struct DataLoader<'a, T, F> {
    data: &'a [T],
    loader: F,
    batchsize: usize,
    count: usize,
}

impl<'a, T, F> DataLoader<'a, T, F> {
    pub fn new(data: &'a [T], loader: F, batchsize: usize) -> Self {
        Self {
            data,
            loader,
            batchsize,
            count: 0,
        }
    }
}

impl<'a, T, F, Feature, Label> Iterator for DataLoader<'a, T, F>
where
    Feature: Element,
    Label: Element,
    F: Fn(&T) -> (Vec<Feature>, Label),
{
    type Item = (Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        if (self.count + 1) * self.batchsize >= self.data.len() {
            return None;
        }
        let mut data = vec![];
        let mut labels = vec![];
        for i in 0..self.batchsize {
            let (x, t) = (self.loader)(&self.data[i + self.count * self.batchsize]);
            data.extend(x);
            labels.push(t);
        }

        self.count += 1;
        Some((Tensor::of_slice(&data), Tensor::of_slice(&labels)))
    }
}
